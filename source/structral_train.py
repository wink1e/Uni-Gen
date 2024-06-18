import torch
import numpy as np
import os
import transform
import get_model
from unicore.data import Dictionary
from active import get_dataloader_active, active_train
from train_model import batch_collate_fn

# Data input
def pad_data(input_xyz_dir):
    r"""
    this function read from carbon cluster data to model input 
    """
    struct_inputs = {'atoms': [], 'coordinates': []}
    struct_prompts = {'atoms': [], 'coordinates': []}
    struct_targets = {'atoms': [], 'coordinates': []}
    for input_file in os.listdir(input_xyz_dir):
        file_path = os.path.join(input_xyz_dir, input_file)
        to_obj = transform.MyMolecule.load_from_xyz(file_path=file_path)
        atom_num = to_obj.atom_num
        for ion in ["Li", "K"]:
            chemical_form = f"{ion}{str(atom_num)}U{str(atom_num)}O{str(5*atom_num)}"
            to_obj.convert(chemical_form=chemical_form)
            new_atom_num = len(to_obj.new_atom_type)
            prompt_atom = to_obj.atom_type
            struct_inputs['atoms'].append(list(prompt_atom)+['[PAD]']*(new_atom_num-atom_num))
            prompt_coord = to_obj.coord
            prompt_coord = np.vstack((prompt_coord, np.zeros(((new_atom_num-atom_num),3))))
            struct_inputs['coordinates'].append(list(prompt_coord))
            struct_prompts['atoms'].append(list(to_obj.new_atom_type))
            struct_prompts['coordinates'].append(list(np.random.random((new_atom_num, 3))))
            struct_targets['atoms'].append(list(to_obj.new_atom_type))
            struct_targets['coordinates'].append(list(to_obj.new_coord))
    return struct_inputs, struct_prompts, struct_targets


if __name__ == "__main__":
    text_path = r'/root/WORK/libin/UniGen/source/mol.dict.txt'
    checkpoint_dir = r'/root/WORK/libin/chk'
    # model = get_model.load_model(text_path, checkpoint_dir)
    dictionary_test = Dictionary.load(text_path)
    d_input, d_prompt, d_target = pad_data(r'/root/WORK/libin/UniGen/source/test1')
    d_train, d_val = get_dataloader_active(d_input, d_prompt, d_target, dictionary_test, batch_size=2, bn_collate_fn=batch_collate_fn)
    model = get_model.load_model(text_path, checkpoint_dir)
    _ = active_train(model, 1e-3, 3, 10, d_train, d_val, device='cuda:0', checkpoint_dir=checkpoint_dir, checkpoint_interval=10)