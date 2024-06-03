import os.path
import re
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import distance_matrix
from unicore.data import Dictionary
from unimol.data.conformer import ConformerGen
from unimol.models.nnmodel import NNDataset
from unimol.tasks.trainer import NNDataLoader
from unimol.utils import pad_coords, pad_2d, pad_1d_tokens

from . import get_data
from . import get_model
from . import randomize_mol


def batch_collate_fn(samples):
    batch = {}
    for k in samples[0][0].keys():
        v = None
        if k.endswith('src_coord'):
            v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
        elif k.endswith('src_edge_type'):
            v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=0.0)
        elif k.endswith('src_distance'):
            v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
        elif k.endswith('src_tokens'):
            v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=0.0)
        assert v is not None
        batch[k] = v
    try:
        label = torch.tensor([s[1] for s in samples])
    except:
        label = None
    return batch, label


# epoch: int, num_epoch: int just for print in the last
# dictionary: only used in random shift
def fit_each_epoch(model: torch.nn.Module, dataloader_train: NNDataLoader,
                   epoch: int, num_epoch: int, dictionary,
                   criterion: torch.nn.Module, device: str,
                   optimizer: torch.optim.Optimizer):
    epoch_begin = time.time()
    model = model.train()
    num_batch = len(dataloader_train)
    loss_data = 0
    loss_baseline = 0
    for batch_i, batch in enumerate(dataloader_train):
        batch_begin = time.time()
        net_input, _ = batch
        # target data after perturbation
        data_perturbation = {'input_src_tokens': deepcopy(net_input['target_src_tokens']),
                             'input_src_distance': deepcopy(net_input['target_src_distance']),
                             'input_src_edge_type': deepcopy(net_input['target_src_edge_type']),
                             'input_src_coord': deepcopy(net_input['target_src_coord'])}

        for i, (atom_i, coord_i) in enumerate(
                zip(data_perturbation['input_src_tokens'], data_perturbation['input_src_coord'])):
            max_id = len(atom_i) - 1
            end_id = np.argwhere(atom_i.numpy() == dictionary.index('[SEP]'))[0, 0]
            gmolecule = randomize_mol.GMolecule(coord=coord_i[1:end_id].view(1, -1).numpy(),
                                                atom_type=np.array([dictionary.symbols[x] for x in atom_i[1:end_id]]))
            gmolecule.random_shift(angle_range=(0, 60), distance=1.0, min_coord_num=3)
            coord_i, _ = gmolecule.get_frame(idx=0)
            data_perturbation['input_src_coord'][i] = torch.from_numpy(
                np.concatenate([np.zeros((1, 3)), coord_i] + [np.zeros((1, 3)) for _ in range(max_id - end_id + 1)],
                               axis=0))
            data_perturbation['input_src_distance'][i] = torch.from_numpy(
                distance_matrix(data_perturbation['input_src_coord'][i], data_perturbation['input_src_coord'][i]))

        model.to(device)
        decoder_distance, decoder_coord = model(net_input['prompt_src_tokens'].to(device),
                                                net_input['prompt_src_distance'].to(device),
                                                net_input['prompt_src_edge_type'].to(device),
                                                net_input['prompt_src_coord'].to(device),
                                                data_perturbation['input_src_tokens'].to(device),
                                                data_perturbation['input_src_distance'].to(device),
                                                data_perturbation['input_src_edge_type'].to(device),
                                                data_perturbation['input_src_coord'].to(device)
                                                )
        loss = criterion(decoder_distance, net_input['target_src_distance'].to(device)) + \
               2 * criterion(decoder_coord, net_input['target_src_coord'].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_data += loss.data
        show_batch_loss_data = loss.data

        loss_baseline += criterion(data_perturbation['input_src_distance'].to(device).data,
                                   net_input['target_src_distance'].to(device).data) + \
                         2 * criterion(data_perturbation['input_src_coord'].to(device).data,
                                       net_input['target_src_coord'].to(device).data)

        batch_time = time.time() - batch_begin
        print('\rBatch {:d}/{:d} Train Loss: {:.8f}, Time: {:.2f}s'.format(batch_i, num_batch, show_batch_loss_data,
                                                                           batch_time), end='', flush=True)
    loss_data /= num_batch
    loss_baseline /= num_batch
    ######################################################achieve this outside
    # loss_train_list.append(float(loss_data.to('cpu')))
    epoch_time = time.time() - epoch_begin
    print('\rEpoch {:d}/{:d} Train Loss: {:.8f}, Baseline Loss: {:.8f}, Time: {:.2f}s, lr: {:.2e}'.format(epoch,
                                                                                                          num_epoch,
                                                                                                          loss_data,
                                                                                                          loss_baseline,
                                                                                                          epoch_time,
                                                                                                          optimizer.state_dict()[
                                                                                                              'param_groups'][
                                                                                                              0]['lr']),
          end='\t')
    #####################################################achieve log later
    # logger.info('Epoch {:d}/{:d} Train Loss: {:.8f}, Time: {:.2f}s'.format(epoch, num_epoch, loss_data, epoch_time))
    return loss_data.to('cpu'), decoder_coord.to('cpu')


# epoch: int, num_epoch: int just for print in the last
def predict_each_epoch(model: torch.nn.Module, dataloader_predict: NNDataLoader,
                       epoch: int, num_epoch: int, dictionary,
                       criterion: torch.nn.Module, device: str):
    model.eval()
    epoch_begin = time.time()
    num_batch = len(dataloader_predict)
    loss_data = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader_predict):
            batch_begin = time.time()
            net_input, _ = batch
            # target data after perturbation
            data_perturbation = {'input_src_tokens': deepcopy(net_input['target_src_tokens']),
                                 'input_src_distance': deepcopy(net_input['target_src_distance']),
                                 'input_src_edge_type': deepcopy(net_input['target_src_edge_type']),
                                 'input_src_coord': deepcopy(net_input['target_src_coord'])}
            for i, (atom_i, coord_i) in enumerate(
                    zip(data_perturbation['input_src_tokens'], data_perturbation['input_src_coord'])):
                max_id = len(atom_i) - 1
                end_id = np.argwhere(atom_i.numpy() == dictionary.index('[SEP]'))[0, 0]
                gmolecule = randomize_mol.GMolecule(coord=coord_i[1:end_id].view(1, -1).numpy(),
                                                    atom_type=np.array(
                                                        [dictionary.symbols[x] for x in atom_i[1:end_id]]))
                gmolecule.random_shift(angle_range=(0, 60), distance=1.0, min_coord_num=3)
                coord_i, _ = gmolecule.get_frame(idx=0)
                data_perturbation['input_src_coord'][i] = torch.from_numpy(
                    np.concatenate([np.zeros((1, 3)), coord_i] + [np.zeros((1, 3)) for _ in range(max_id - end_id + 1)],
                                   axis=0))
                # ([np.zeros((1, 3)), coord_i] + [np.zeros((1, 3) for _ in range(max_id - end_id)]), axis=0))
                # only one frame in gmolecule
                data_perturbation['input_src_distance'][i] = torch.from_numpy(
                    distance_matrix(data_perturbation['input_src_coord'][i], data_perturbation['input_src_coord'][i]))

            model.to(device)
            decoder_distance, decoder_coord = model(net_input['prompt_src_tokens'].to(device),
                                                    net_input['prompt_src_distance'].to(device),
                                                    net_input['prompt_src_edge_type'].to(device),
                                                    net_input['prompt_src_coord'].to(device),
                                                    data_perturbation['input_src_tokens'].to(device),
                                                    data_perturbation['input_src_distance'].to(device),
                                                    data_perturbation['input_src_edge_type'].to(device),
                                                    data_perturbation['input_src_coord'].to(device)
                                                    )
            loss = criterion(decoder_distance, net_input['target_src_distance'].to(device)) + \
                   2 * criterion(decoder_coord, net_input['target_src_coord'].to(device))
            loss_data += loss.data
            show_batch_loss_data = loss.data
            batch_time = time.time() - batch_begin
            # print('\rBatch {:d}/{:d} Validation Loss: {:.8f}, Time: {:.2f}s'.format(batch_i, num_batch, show_batch_loss_data, batch_time), end='', flush=True)
    loss_data /= num_batch
    ##############################################################achieve this outside
    # loss_val_list.append(float(loss_data.to('cpu')))
    epoch_time = time.time() - epoch_begin
    print('Epoch {:d}/{:d} Validation Loss: {:.8f}, Time: {:.2f}s'.format(epoch, num_epoch, loss_data, epoch_time))
    # print('\rEpoch {:d}/{:d} Validation Loss: {:.8f}, Time: {:.2f}s'.format(epoch, num_epoch, loss_data, epoch_time))
    # logger.info('Epoch {:d}/{:d} Validation Loss: {:.8f}, Time: {:.2f}s'.format(epoch, num_epoch, loss_data, epoch_time))
    return loss_data.to('cpu'), decoder_coord.to('cpu')


def get_dataloader(data_target, data_prompt, dictionary, batch_size: int,
                   batch_collate_fn=batch_collate_fn):
    conformergen = ConformerGen()
    conformergen.dictionary = dictionary
    data_target = conformergen.transform_raw(data_target['atoms'], data_target['coordinates'])
    data_target = np.asarray(data_target)
    # prompt data
    data_prompt = conformergen.transform_raw(data_prompt['atoms'], data_prompt['coordinates'])
    # final data
    assert len(data_prompt) == len(
        data_target), f'len(data_prompt)={len(data_prompt)}  len(data_target)={len(data_target)}'
    data = np.empty(shape=data_target.shape, dtype=dict)
    for i in range(len(data)):
        data[i] = {}
        data[i]['prompt_src_tokens'] = data_prompt[i]['src_tokens']
        data[i]['prompt_src_distance'] = data_prompt[i]['src_distance']
        data[i]['prompt_src_edge_type'] = data_prompt[i]['src_edge_type']
        data[i]['prompt_src_coord'] = data_prompt[i]['src_coord']
        data[i]['target_src_tokens'] = data_target[i]['src_tokens']
        data[i]['target_src_distance'] = data_target[i]['src_distance']
        data[i]['target_src_edge_type'] = data_target[i]['src_edge_type']
        data[i]['target_src_coord'] = data_target[i]['src_coord']
    dataset = NNDataset(data)
    dataloader = NNDataLoader(
        feature_name=None,
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_collate_fn,
    )
    return dataloader


text_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mol.dict.txt')  # 原子类型映射词表的存储位置


def train_model(lr: float, lr_patience: int,
                num_epoch: int, batch_size: int,
                pkl_dir: str, device: str,
                checkpoint_dir: str, checkpoint_interval: int,
                if_read_checkpoint: bool):
    dict_temp = Dictionary.load(text_path)

    # define dataset
    # target data
    data_target_train, data_prompt_train = get_data.get_data_from_pkl(load_path=pkl_dir, if_train=True, if_val=False)
    data_target_val, data_prompt_val = get_data.get_data_from_pkl(load_path=pkl_dir, if_train=False, if_val=True)

    dataloader_train = get_dataloader(data_target=data_target_train,
                                      data_prompt=data_prompt_train,
                                      dictionary=dict_temp, batch_size=batch_size,
                                      batch_collate_fn=batch_collate_fn)
    dataloader_val = get_dataloader(data_target=data_target_val,
                                    data_prompt=data_prompt_val,
                                    dictionary=dict_temp, batch_size=batch_size,
                                    batch_collate_fn=batch_collate_fn)

    # initialize loss, optimizer, model, loss list
    models = get_model.define_model(text_path=text_path)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(models.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience)
    loss_train_list = []  # Add
    loss_val_list = []

    # initialize epoch, optimizer, model, loss list from last checkpoint(if exist)
    ###############################################################################import valuable used in the training process
    start_epoch = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    if if_read_checkpoint:
        for checkpoint in os.listdir(checkpoint_dir):
            # Error checkpoint = os.path.join(checkpoint_dir, checkpoint)
            if checkpoint.endswith('.pkl') and ('checkpoint' in checkpoint):
                start_epoch_new = re.findall(r'(\d+)', checkpoint)
                assert len(start_epoch_new) == 1, f'wrong checkpoint file name: {checkpoint}'
                start_epoch = max(start_epoch, int(start_epoch_new[0]))
        if start_epoch != 0:  # checkpoint file exist
            ############################################################################important default setting name of checkpoint file
            checkpoint = os.path.join(checkpoint_dir, f'checkpoint_{start_epoch}_epoch.pkl')
            # Error Mistake checkpoint = os.path.join(checkpoint_dir, f'checkpoint_{start_epoch_new}_epoch.pkl')
            checkpoint = torch.load(checkpoint)
            # checkpoint = torch.load(checkpoint, map_location=device) # change from path(str) to checkpoint(torch.class)
            models.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            assert start_epoch == checkpoint[
                'epoch'], f"start_epoch={start_epoch}  checkpoint['epoch']={checkpoint['epoch']}"
            optimizer = torch.optim.Adam(models.parameters(), lr=optimizer.state_dict()['param_groups'][0]['lr'])
            loss_train_list = checkpoint['loss_train_list']
            loss_val_list = checkpoint['loss_val_list']
    # define training process
    assert start_epoch < num_epoch, f'start_epoch={start_epoch}  num_epoch={num_epoch}'

    for epoch in range(start_epoch, num_epoch):  # epoch start from 0
        loss_train, _ = fit_each_epoch(model=models, dataloader_train=dataloader_train,
                                       epoch=epoch, num_epoch=num_epoch, dictionary=dict_temp,
                                       criterion=criterion, device=device,
                                       optimizer=optimizer)
        loss_val, _ = predict_each_epoch(model=models, dataloader_predict=dataloader_val,
                                         epoch=epoch, num_epoch=num_epoch, dictionary=dict_temp,
                                         criterion=criterion, device=device)
        scheduler.step(loss_val)
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        # save checkpoint file
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": models.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "loss_train_list": loss_train_list,
                          "loss_val_list": loss_val_list}
            path_checkpoint = os.path.join(checkpoint_dir, "checkpoint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

    assert len(loss_train_list) == len(loss_val_list)

    plt.rcParams.update({'font.size': 23})
    xs = np.arange(len(loss_train_list))
    ys = np.array(loss_train_list)
    zs = np.array(loss_val_list)
    print(xs)
    print(ys)
    print(zs)
    plt.plot(xs, ys, alpha=0.7, linewidth=10, label='Train Loss')
    plt.plot(xs, zs, alpha=0.7, linewidth=10, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAELoss')
    plt.legend(prop={'size': 23})
    plt.show()
