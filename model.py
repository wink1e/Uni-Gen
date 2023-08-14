import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from unicore import utils
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from unicore.modules import SelfMultiheadAttention, softmax_dropout

from .unimol_source import UniMolModel

logger = logging.getLogger()


@register_model("unimolgen")
class UniMolGen(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_model(cls, args, dictionary):
        return cls(args, dictionary)

    def __init__(self, args: argparse.Namespace(), dictionary: Dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.encoder = UniMolModel(args.encoder, dictionary)
        self.decoder = GenDecoder(args.decoder, dictionary)

    def forward(
            self,
            prompt_src_tokens: Tensor,
            prompt_src_distance: Tensor,
            prompt_src_edge_type: Tensor,
            prompt_src_coord: Tensor,
            input_src_tokens: Tensor,
            input_src_distance: Tensor,
            input_src_edge_type: Tensor,
            input_src_coord: Tensor,
            encoder_masked_tokens=None,
            features_only=False,
            classification_head_name=None,
            **kargs
    ) -> Tensor:
        encoder_rep, pair_rep = self.encoder(prompt_src_tokens, prompt_src_distance, prompt_src_coord,
                                             prompt_src_edge_type)

        decoder_distance, decoder_coord = self.decoder(input_src_tokens, input_src_distance,
                                                       input_src_coord, input_src_edge_type,
                                                       encoder_rep, pair_rep, )

        return decoder_distance, decoder_coord


class GenDecoder(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()

        self.decoder_layers = args.decoder_layers
        self.embed_dim = args.embed_dim
        self.ffn_embed_dim = args.ffn_embed_dim
        self.attention_heads = args.attention_heads
        self.emb_dropout = args.emb_dropout
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.max_seq_len = args.max_seq_len
        self.activation_fn = args.activation_fn

        self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.final_head_layer_norm = LayerNorm(self.attention_heads)
        self.embed_tokens = nn.Embedding(len(dictionary), self.embed_dim, self.padding_idx)
        self.gbf_proj = NonLinearHead(128, self.attention_heads, self.activation_fn)
        self.gbf = GaussianLayer(128, len(dictionary) ** 2)
        self.pair2coord_proj = NonLinearHead(self.attention_heads, 1, self.activation_fn)
        self.dist_head = DistanceHead(self.attention_heads, self.activation_fn)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=self.embed_dim,
                ffn_embed_dim=self.ffn_embed_dim,
                attention_heads=self.attention_heads,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation_dropout=self.activation_dropout,
                activation_fn=self.activation_fn,
                post_ln=False
            )
            for l in range(self.decoder_layers)
        ])
        self.apply(init_bert_params)

    def forward(
            self,
            input_src_tokens: Tensor,
            input_src_distance: Tensor,
            input_src_coord: Tensor,
            input_src_edge_type: Tensor,
            encoder_rep: Tensor,
            pair_rep: Tensor,
    ) -> Tensor:
        padding_mask = input_src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(input_src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(input_src_distance, input_src_edge_type)

        bsz, seq_len = [x.size(i) for i in (0, 1)]
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = graph_attn_bias
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert graph_attn_bias is not None
        graph_attn_bias, padding_mask = fill_attn_mask(graph_attn_bias, padding_mask)

        for layer in self.layers:
            x, graph_attn_bias, pair_rep = layer(x, encoder_out=encoder_rep,
                                                 attn_bias=graph_attn_bias,
                                                 padding_mask=padding_mask,
                                                 encoder_attn_bias=pair_rep,
                                                 encoder_padding_mask=None,
                                                 return_attn=True)

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x ** 2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                    torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = graph_attn_bias - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        graph_attn_bias = (
            graph_attn_bias.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        graph_attn_bias[graph_attn_bias == float("-inf")] = 0
        decoder_distance = None
        decoder_coord = None
        if self.args.masked_coord_loss > 0:
            padding_mask = input_src_tokens.eq(self.padding_idx)
            if not padding_mask.any():
                padding_mask = None
            if padding_mask is not None:
                atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(-1, 1, 1, 1)
            else:
                atom_num = input_src_coord.shape[1] - 1
            delta_pos = input_src_coord.unsqueeze(1) - input_src_coord.unsqueeze(2)
            attn_probs = self.pair2coord_proj(delta_pair_repr)
            coord_update = delta_pos / atom_num * attn_probs
            coord_update = torch.sum(coord_update, dim=2)
            decoder_coord = input_src_coord + coord_update
        if self.args.masked_dist_loss > 0:
            decoder_distance = self.dist_head(graph_attn_bias)

        return decoder_distance, decoder_coord


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
            self,
            input_dim,
            out_dim,
            activation_fn,
            hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
            self,
            heads,
            activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class TransformerDecoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer with attn returned.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            ffn_embed_dim: int = 3072,
            attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.0,
            activation_fn: str = "gelu",
            post_ln=False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(
            self,
            x: torch.Tensor,
            encoder_out: torch.Tensor = None,
            attn_bias: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            encoder_attn_bias: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            return_attn: bool = False
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        if encoder_out is not None:
            residual = x
            if not self.post_ln:
                x = self.encoder_attn_layer_norm(x)
            x = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                attn_bias=encoder_attn_bias,
                return_attn=return_attn
            )
            # x = self.dropout_module(x)
            if return_attn:
                x, encoder_attn_weights, encoder_attn_probs = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if self.post_ln:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if return_attn:
            return x, attn_weights, encoder_attn_weights
        else:
            return x


class CrossMultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.1,
            bias=True,
            scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask: Optional[Tensor] = None,
            attn_bias: Optional[Tensor] = None,
            return_attn: bool = False
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = (
                q.view(bsz, tgt_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
                * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = softmax_dropout(attn_weights, self.dropout, self.training, bias=attn_bias)
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(attn_weights, self.dropout, self.training, inplace=False)

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if return_attn:
            return o, attn_weights, attn
        else:
            return o


@register_model_architecture("unimolgen", "unimolgen")
def base_architecture(args):
    args.encoder.encoder_layers = getattr(args.encoder, "encoder_layers", 6)
    args.encoder.encoder_embed_dim = getattr(args.encoder, "encoder_embed_dim", 512)
    args.encoder.encoder_ffn_embed_dim = getattr(args.encoder, "encoder_ffn_embed_dim", 2048)
    args.encoder.encoder_attention_heads = getattr(args.encoder, "encoder_attention_heads", 64)
    args.encoder.dropout = getattr(args.encoder, "dropout", 0.1)
    args.encoder.emb_dropout = getattr(args.encoder, "emb_dropout", 0.1)
    args.encoder.attention_dropout = getattr(args.encoder, "attention_dropout", 0.1)
    args.encoder.activation_dropout = getattr(args.encoder, "activation_dropout", 0.0)
    args.encoder.pooler_dropout = getattr(args.encoder, "pooler_dropout", 0.0)
    args.encoder.max_seq_len = getattr(args.encoder, "max_seq_len", 512)
    args.encoder.activation_fn = getattr(args.encoder, "activation_fn", "gelu")
    args.encoder.pooler_activation_fn = getattr(args.encoder, "pooler_activation_fn", "tanh")
    args.encoder.post_ln = getattr(args.encoder, "post_ln", False)
    args.encoder.masked_token_loss = getattr(args.encoder, "masked_token_loss", -1.0)
    args.encoder.masked_coord_loss = getattr(args.encoder, "masked_coord_loss", -1.0)
    args.encoder.masked_dist_loss = getattr(args.encoder, "masked_dist_loss", -1.0)
    args.encoder.x_norm_loss = getattr(args.encoder, "x_norm_loss", -1.0)
    args.encoder.delta_pair_repr_norm_loss = getattr(args.encoder, "delta_pair_repr_norm_loss", -1.0)
    args.encoder.mode = getattr(args.encoder, "mode", "infer")

    args.decoder.decoder_layers = getattr(args.decoder, "decoder_layers", 6)
    args.decoder.embed_dim = getattr(args.decoder, "decoder_embed_dim", 512)
    args.decoder.ffn_embed_dim = getattr(args.decoder, "decoder_ffn_embed_dim", 2048)
    args.decoder.attention_heads = getattr(args.decoder, "decoder_attention_heads", 64)
    args.decoder.dropout = getattr(args.decoder, "dropout", 0.1)
    args.decoder.emb_dropout = getattr(args.decoder, "emb_dropout", 0.1)
    args.decoder.attention_dropout = getattr(args.decoder, "attention_dropout", 0.1)
    args.decoder.activation_dropout = getattr(args.decoder, "activation_dropout", 0.0)
    args.decoder.pooler_dropout = getattr(args.decoder, "pooler_dropout", 0.0)
    args.decoder.max_seq_len = getattr(args.decoder, "max_seq_len", 512)
    args.decoder.activation_fn = getattr(args.decoder, "activation_fn", "gelu")
    args.decoder.pooler_activation_fn = getattr(args.decoder, "pooler_activation_fn", "tanh")
    args.decoder.post_ln = getattr(args.decoder, "post_ln", False)
    args.decoder.masked_coord_loss = getattr(args.decoder, "masked_coord_loss", 1.0)
    args.decoder.masked_dist_loss = getattr(args.decoder, "masked_dist_loss", 1.0)


@register_model_architecture("unimolgen", "unimolgen_base")
def unimol_base_architecture(args):
    base_architecture(args)


if __name__ == '__main__':
    model = UniMolGen()
    print("Hello world!")
