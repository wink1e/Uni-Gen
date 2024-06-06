import os
from copy import deepcopy

import numpy as np
import torch
from scipy.spatial import distance_matrix
from unicore.data import Dictionary
from unimol_tools.data.conformer import ConformerGen
from unimol_tools.models.nnmodel import NNDataset
from unimol_tools.tasks.trainer import NNDataLoader
from unimol_tools.utils import pad_coords, pad_2d, pad_1d_tokens
import randomize_mol
import get_model
import re
import tqdm


def xyz_to_list(filename: str):
    r"""
    Read xyz file of one frame(can be improved)

    Returns:

    atoms: 1d-list of string, shape=(natoms,), such as
        ['Na', 'Na']

    coordinates: 2d-list of float64, shape=(natoms, 3), such as
        [[-4.391989, 0.919716, 1.48585], [3.477594, 0.400666, 2.863943]]
    """
    '''Test code
    #######################################################Linux
    dir_path = os.path.abspath('./source')
    xyz_file_path = os.path.join(dir_path, 'UO2_2_O2_4_OH_2_Na_6_another_from3.xyz')
    atoms, coordinates = read_one_xyz(filename=xyz_file_path)
    print(atoms)
    print(coordinates)
    '''
    assert os.path.exists(filename), f'There is NO xyz file of {os.path.abspath(filename)}'
    natom = 0
    atoms = []
    coordinations = []

    with open(filename, 'r') as f:
        natom = int(f.readline().split()[0])
        _ = f.readline()
        for line in f.readlines():
            line = line.split()
            if len(line) == 0:
                break
            else:
                atoms.append(line[0])
                coordinations.append([float(line[i]) for i in range(1, 4)])  # float64
    assert len(atoms) == natom, f'The xyz file is broken. File path: {os.path.abspath(filename)}'
    return atoms, coordinations


def list_to_unimol_input_dict(atoms, coordinations, energies=[]):
    r"""
    Get Unimol Input Dict of one frame(can be improved)

    Parameters:

    atoms: 1d-list of string, shape=(natoms,), such as
        ['Na', 'Na']

    coordinates: 2d-list of float64, shape=(natoms, 3), such as
        [[-4.391989, 0.919716, 1.48585], [3.477594, 0.400666, 2.863943]]

    Returns:

    unimol_input_dict: dictionary with keys 'atoms', 'coordinates', 'target'(optional),
        with values unimol_input_dict['atoms']: 1d-list of (1d-np.ndarray of string), shape=(nframe, (natoms,))
        with values unimol_input_dict['coordinates']: 1d-list of (2d-np.ndarray of float), shape=(nframe, (natoms, 3))
        with values(optional) unimol_input_dict['target']: 1d-list of (1d-np.ndarray of float), shape=(nframe, (1,))

        such as
        {'atoms': [np.array('Na', 'Na'), np.array('Na', 'Na')]
        'coordinates': [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        'target': []}
    """
    nframe = 1
    unimol_input_dict = {'atoms': [], 'coordinates': [], 'target': []}
    # atom type should be the same!!!!!!!!!!!!!!!!!!!!
    # change if necessary
    unimol_input_dict['atoms'] += [np.array(atoms) for _ in range(nframe)]
    unimol_input_dict['coordinates'] += list(np.array(coordinations).reshape(nframe, -1, 3))
    if energies != []:
        unimol_input_dict['target'] += list(np.array(energies).reshape(nframe, 1))
    return unimol_input_dict


def unimol_input_dict_to_unimol_model_dict(unimol_input_dict, text_path: str):
    r"""
    Get Unimol Model Dict (no need to improve)

    Patameters:

    unimol_input_dict: dictionary with keys 'atoms', 'coordinates', 'target'(optional),
        with values unimol_input_dict['atoms']: 1d-list of (1d-np.ndarray of string), shape=(nframe, (natoms,))
        with values unimol_input_dict['coordinates']: 1d-list of (2d-np.ndarray of float), shape=(nframe, (natoms, 3))
        with values(optional) unimol_input_dict['target']: 1d-list of (1d-np.ndarray of float), shape=(nframe, (1,))

        such as
        {'atoms': [np.array('Na', 'Na'), np.array('Na', 'Na')]
        'coordinates': [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        'target': []}

    text_path: path of dictionary embedding txt

    Returns:

    unimol_model_dict: 1d-np.ndarray of
        (dictionary with keys 'src_tokens', 'src_distance', 'src_coord',  'src_edge_type'), shape=(nframe,)
    """
    conformergen = ConformerGen()
    dictionary = Dictionary.load(text_path)
    conformergen.dictionary = dictionary
    unimol_model_dict = conformergen.transform_raw(unimol_input_dict['atoms'], unimol_input_dict['coordinates'])
    unimol_model_dict = np.asarray(unimol_model_dict)
    return unimol_model_dict


def get_dataloader(unimol_model_dict_prompt, unimol_model_dict_input, batch_size: int, if_shuffle: bool):
    r"""
    Get dataloader of out model from 2 unimol_model_dict(prompt and input)
    Note that when testing, these two dicts are the same

    Parameters:

    unimol_model_dict_prompt, unimol_model_dict_input:
        Both are unimol_model_dict:
        1d-np.ndarray of (dictionary with keys 'src_tokens', 'src_distance', 'src_coord',  'src_edge_type'), shape=(nframe,)

    Returns:

    dataloader: iterable object of dataloader_dict:
        Dictionary with keys 'prompt_src_tokens', 'prompt_src_distance', 'prompt_src_edge_type',
        'prompt_src_coord', 'target_src_tokens', 'target_src_distance', 'target_src_edge_type', 'target_src_coord'
        ), Notice that shape of each value=(batch_size,)

    """

    def batch_collate_fn(samples):
        batch = {}
        for k in samples[0][0].keys():
            v = None  # add
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

    assert len(unimol_model_dict_prompt) == len(unimol_model_dict_input), \
        f'len(unimol_model_dict_prompt)={len(unimol_model_dict_prompt)}  ' \
        f'len(unimol_model_dict_input)={len(unimol_model_dict_input)}'
    data = np.empty(shape=unimol_model_dict_input.shape, dtype=dict)
    for i in range(len(data)):
        data[i] = {}
        data[i]['prompt_src_tokens'] = unimol_model_dict_prompt[i]['src_tokens']
        data[i]['prompt_src_distance'] = unimol_model_dict_prompt[i]['src_distance']
        data[i]['prompt_src_edge_type'] = unimol_model_dict_prompt[i]['src_edge_type']
        data[i]['prompt_src_coord'] = unimol_model_dict_prompt[i]['src_coord']
        data[i]['target_src_tokens'] = unimol_model_dict_input[i]['src_tokens']
        data[i]['target_src_distance'] = unimol_model_dict_input[i]['src_distance']
        data[i]['target_src_edge_type'] = unimol_model_dict_input[i]['src_edge_type']
        data[i]['target_src_coord'] = unimol_model_dict_input[i]['src_coord']
    dataset = NNDataset(data)
    dataloader = NNDataLoader(
        feature_name=None,
        dataset=dataset,
        batch_size=batch_size,
        shuffle=if_shuffle,
        collate_fn=batch_collate_fn,
    )
    return dataloader


def predict(model, dataloader, text_path: str,
            angle_range, distance: float, min_coord_num: float,
            device='cpu'):
    r"""
    Get one unimol_model_dict(target)

    Parameters:

    dataloader: iterable object of dataloader_dict:
        Dictionary with keys 'prompt_src_tokens', 'prompt_src_distance', 'prompt_src_edge_type',
        'prompt_src_coord', 'target_src_tokens', 'target_src_distance', 'target_src_edge_type', 'target_src_coord'
        ), Notice that shape of each value=(batch_size,)

    Returns:

    data_output: unimol_model_dict:
        1d-np.ndarray of (dictionary with keys 'src_tokens', 'src_distance', 'src_coord',
        'src_edge_type'), shape=(nbatch,)

    """
    dictionary = Dictionary.load(text_path)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            net_input, _ = batch
            break
        batch_size = len(net_input['target_src_tokens'])
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
            gmolecule.random_shift(angle_range=angle_range, distance=distance, min_coord_num=min_coord_num)
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
    data_random = [{'src_tokens': data_perturbation['input_src_tokens'][i],
                    'src_distance': data_perturbation['input_src_distance'][i],
                    'src_coord': data_perturbation['input_src_coord'][i],
                    'src_edge_type': data_perturbation['input_src_edge_type'][i]} for i in range(batch_size)]

    data_output = [{'src_tokens': data_perturbation['input_src_tokens'][i],
                    'src_distance': decoder_distance[i],
                    'src_coord': decoder_coord[i],
                    'src_edge_type': data_perturbation['input_src_edge_type'][i]} for i in range(batch_size)]
    return np.asarray(data_output), np.asarray(data_random)


def unimol_model_dict_to_list(unimol_model_dict, text_path: str):
    r"""
    Read from unimol_model

    Parameters:

    unimol_model_dict:
        1d-np.ndarray of (dictionary with keys 'src_tokens', 'src_distance', 'src_coord',
        'src_edge_type'), shape=(nframe,) # Or (batch_size, )

    Returns:

    atoms: 2d-list of string, shape=(nframe, natoms,), such as
        [['Na', 'Na'], ['Na', 'Na']]

    coordinates: 3d-list of float64, shape=(nframe, natoms, 3), such as
        [ [[-4.391989, 0.919716, 1.48585], [3.477594, 0.400666, 2.863943]],
         [[-4.391989, 0.919716, 1.48585], [3.477594, 0.400666, 2.863943]] ]

    """
    dictionary = Dictionary.load(text_path)
    atoms = []
    coordinates = []
    for i in range(len(unimol_model_dict)):
        atom_i = unimol_model_dict[i]['src_tokens']
        coord_i = unimol_model_dict[i]['src_coord']
        max_id = len(atom_i) - 1
        end_id = np.argwhere(atom_i.numpy() == dictionary.index('[SEP]'))[0, 0]
        atoms.append([dictionary.symbols[x] for x in atom_i[1:end_id]])
        coordinates.append(coord_i[1:end_id].view(-1, 3))
    return atoms, coordinates


def list_to_xyz(filename: str, atom_i, coord_i):
    r"""
    write xyz from one frame into xyz_save_path/xyz_filename
    """
    with open(filename, 'w') as f:
        f.write(f'{int(len(atom_i))}\n\n')
        for j in range(len(atom_i)):
            f.write(f'{atom_i[0][j]}\t{float(coord_i[0][j][0])}\t'
                    f'{float(coord_i[0][j][1])}\t'
                    f'{float(coord_i[0][j][2])}\n')


def initialize_model(if_read_checkpoint: bool, checkpoint_dir: str, text_path: str):
    model = get_model.define_model(text_path=text_path)
    start_epoch = 0
    if if_read_checkpoint:
        for checkpoint in os.listdir(checkpoint_dir):
            if checkpoint.endswith('.pkl') and ('checkpoint' in checkpoint):
                start_epoch_new = re.findall(r'(\d+)', checkpoint)
                assert len(start_epoch_new) == 1, f'wrong checkpoint file name: {checkpoint}'
                start_epoch = max(start_epoch, int(start_epoch_new[0]))
    if start_epoch != 0:  # checkpoint file exist
        checkpoint = os.path.join(checkpoint_dir, f'checkpoint_{start_epoch}_epoch.pkl')
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        assert start_epoch == checkpoint[
            'epoch'], f"start_epoch={start_epoch}  checkpoint['epoch']={checkpoint['epoch']}"
    return model


text_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mol.dict.txt')  # 原子类型映射词表的存储位置


def test_model(input_dir: str, output_dir: str, checkpoint_dir: str,
               angle_range, distance: float, min_coord_num: float,
               device: str):
    os.makedirs(output_dir, exist_ok=True)
    good_dir = os.path.join(output_dir, 'good_dir')
    bad_dir = os.path.join(output_dir, 'bad_dir')
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    for project in tqdm.tqdm(os.listdir(input_dir)):
        if project.endswith('.xyz'):
            input_xyz_filename = os.path.join(input_dir, project)
            bad_xyz_filename = os.path.join(bad_dir, project)
            good_xyz_filename = os.path.join(good_dir, project)

            atoms, coordinations = xyz_to_list(filename=input_xyz_filename)
            unimol_input_dict = list_to_unimol_input_dict(atoms=atoms, coordinations=coordinations, energies=[])
            unimol_model_dict = unimol_input_dict_to_unimol_model_dict(unimol_input_dict=unimol_input_dict,
                                                                       text_path=text_path)
            dataloader = get_dataloader(unimol_model_dict_prompt=unimol_model_dict,
                                        unimol_model_dict_input=unimol_model_dict,
                                        batch_size=1, if_shuffle=False)
            model = initialize_model(if_read_checkpoint=True, checkpoint_dir=checkpoint_dir, text_path=text_path)
            data_output, data_random = predict(model=model, dataloader=dataloader, text_path=text_path,
                                               angle_range=angle_range, distance=distance, min_coord_num=min_coord_num,
                                               device=device)
            atoms, coordinates = unimol_model_dict_to_list(unimol_model_dict=data_random, text_path=text_path)
            list_to_xyz(filename=bad_xyz_filename, atom_i=atoms, coord_i=coordinates)
            atoms, coordinates = unimol_model_dict_to_list(unimol_model_dict=data_output, text_path=text_path)
            list_to_xyz(filename=good_xyz_filename, atom_i=atoms, coord_i=coordinates)
