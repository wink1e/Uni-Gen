import argparse
import os
import torch
import re
from unicore.data import Dictionary
from . import model


def define_model(text_path: str):
    parser = argparse.Namespace()
    parser.encoder = argparse.Namespace()
    parser.decoder = argparse.Namespace()

    dict_temp = Dictionary.load(text_path)
    models = model.UniMolGen(parser, dict_temp)
    return models


def load_model(text_path: str, checkpoint_dir: str):
    model = define_model(text_path=text_path)
    for checkpoint in os.listdir(checkpoint_dir):
        if checkpoint.endswith('.pkl') and 'checkpoint' in checkpoint:
            start_epoch = re.findall(r'(\d+)', checkpoint)
            start_epoch = max(0, int(start_epoch[0]))
    assert start_epoch != 0, "please check the checkpoint directory!"
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{start_epoch}_epoch.pkl')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model