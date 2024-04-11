import math
import torch
import torch.utils.data
import numpy as np
import time
import os
import get_data
import get_model
from unicore.data import Dictionary
try:
    from unimol.data.conformer import ConformerGen
except ModuleNotFoundError:
    from unimol_tools.data.conformer import ConformerGen
# This script is the main program of active learning part.

text_path = os.path.join('../', 'source', 'mol.dict.txt')


def active_learning_iter(old_model, max_iter):
    r"""
    this function is the main active learning iteration process.
    :param old_model:
    :param max_iter:
    :return:
    """
    pass


def read_opt_file(file_path, prompt_path=None):
    r"""
    this function read a cp2k_opt_out traj and convert it to active training data.
    :param file_path:
    :prompt_path:
    :return:
    """
    active_input = {'atoms': [], 'coordinates': [], 'energy': []}
    active_target = {'atoms': [], 'coordinates': [], 'energy': []}
    active_prompt = {'atoms': [], 'coordinates': [], 'energy': []}
    input_index = []
    target_index = []
    traj_all = get_data.CP2K_MD(file_path)
    traj_all.load_traj_all()
    n_frame = len(traj_all.pot)
    old_ener = traj_all.pot[0]
    old_index = 0
    for i in range(n_frame):
        if i == 0:
            ener_diff = 0
        else:
            ener_diff = traj_all.pot[i] - old_ener
        if ener_diff <= -0.05:
            input_index.append(old_index)
            target_index.append(i)
            old_index = i
            old_ener = traj_all.pot[i]
    if len(input_index) == 0:
        return None, None, None
    active_input['atoms'] += [traj_all.atom_type for _ in range(len(input_index))]
    active_input['coordinates'] += list(traj_all.coord[input_index].reshape(len(input_index), -1, 3))
    print(f"Active training data: {len(input_index)} frames")
    if prompt_path is not None:
        prompt = get_data.CP2K_MD(prompt_path)
        prompt.load_traj_all()
        active_prompt['atoms'] += [prompt.atom_type for _ in range(len(input_index))]
        active_prompt['coordinates'] += [prompt.coord.reshape(-1, 3) for _ in range(len(input_index))]
    else:
        prompt_index = np.argmin(traj_all.pot)
        active_prompt['atoms'] += [traj_all.atom_type for _ in range(len(input_index))]
        active_prompt['coordinates'] += [traj_all.coord[prompt_index].reshape(-1, 3) for _ in range(len(input_index))]
    active_target['atoms'] += [traj_all.atom_type for _ in range(len(target_index))]
    active_target['coordinates'] += list(traj_all.coord[target_index].reshape(len(target_index), -1, 3))
    return active_input, active_prompt, active_target


def read_opt_files(dir_path, prompt_path=None):
    active_inputs = {'atoms': [], 'coordinates':[], 'energy': []}
    active_targets = {'atoms': [], 'coordinates':[], 'energy': []}
    active_prompts = {'atoms': [], 'coordinates':[], 'energy': []}
    assert os.path.exists(dir_path)
    for subdir in os.listdir(dir_path):
        file_path = os.path.join(dir_path, subdir)
        if os.path.isdir(file_path):
            print(f'Loading form path {file_path}:')
            temp_input, temp_prompt, temp_target = read_opt_file(file_path=file_path, prompt_path=prompt_path)
            active_inputs['atoms'] += temp_input['atoms']
            active_inputs['coordinates'] += temp_input['coordinates']
            active_targets['atoms'] += temp_target['atoms']
            active_targets['coordinates'] += temp_target['coordinates']
            active_prompts['atoms'] += temp_prompt['atoms']
            active_prompts['coordinates'] += temp_prompt['coordinates']
    return active_inputs, active_prompts, active_targets


def active_train(old_model, lr: float, lr_patience: int, num_epoch: int,
                 train_loader, val_loader, device, checkpoint_dir, checkpoint_interval):
    # initialize training parameter
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(old_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience)
    loss_train_list = []
    loss_val_list = []
    start_epoch = 0
    loss_epoch_train = 0
    loss_epoch_val = 0
    begin_epoch_time = time.time()
    # training process
    model = old_model.to(device)
    for epoch in range(start_epoch, num_epoch):
        # training
        model = model.train()
        num_batch = len(train_loader)
        for batch_i, batch in enumerate(train_loader):
            begin_time = time.time()
            net_input, net_target = batch
            decoder_distance, decoder_coord = model(*net_input)
            loss = (criterion(decoder_distance, net_target[1].to(device))
                    + 2 * criterion(decoder_coord, net_input[3].to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch_train += loss.data
            batch_time = time.time() - begin_time
            print(f"\r Batch {batch_i}/{num_batch}==Train loss {loss.data:.8f}, Time {batch_time:.2f}s",
                  end='', flush=True)
        loss_epoch_train /= num_batch
        epoch_time = time.time() - begin_epoch_time
        print(f"\r Epoch {epoch}/{num_epoch}==Train loss: {loss_epoch_train:.8f}, Time {epoch_time:.2f}", end='\t')

        # validation process
        model.eval()
        num_batch = len(val_loader)
        begin_epoch_time = time.time()
        with torch.no_grad():
            for batch_i, batch in enumerate(val_loader):
                begin_time = time.time()
                net_input, net_target = batch
                decoder_distance, decoder_coord = model(*net_input)
                loss = (criterion(decoder_distance, net_target[1].to(device))
                        + 2 * criterion(decoder_coord, net_input[3].to(device)))
                loss_epoch_val += loss.data
                batch_time = time.time() - begin_time
                print(f"\r Batch {batch_i}/{num_batch}==Validation loss {loss.data:.8f}, Time {batch_time:.2f}s",
                      end='', flush=True)
            loss_epoch_val /= num_batch
            epoch_time = time.time() - begin_epoch_time
            print(f"\r Epoch {epoch}/{num_epoch}==Validation loss: {loss_epoch_train:.8f}, Time {epoch_time:.2f}", end='\t')
        scheduler.step(loss_epoch_val)
        loss_train_list.append(loss_epoch_train)
        loss_val_list.append(loss_epoch_val)

        # Save checkpoint file
        if epoch % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "loss_train_list": loss_train_list,
                          "loss_val_list": loss_val_list}
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_epoch_active.pkl")
            torch.save(checkpoint, checkpoint_path)


def get_dataloader_active(data_input, data_prompt, data_target, dictionary, batch_size):
    dataset = MyDataset(data_input, data_prompt, data_target, dictionary)
    num_set = len(dataset)
    num_train = math.floor(0.8 * num_set)
    num_val = num_set - num_train
    train_set, val_set = torch.utils.data.random_split(dataset=dataset, lengths=[num_train, num_val])
    dataloader_train = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size
    )
    return dataloader_train, dataloader_val


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, prompt_data, target_data, dictionary):
        conformer = ConformerGen()
        conformer.dictionary = dictionary

        self.data_input = conformer.transform_raw(input_data['atoms'], input_data['coordinates'])
        self.data_prompt = conformer.transform_raw(prompt_data['atoms'], prompt_data['coordinates'])
        self.data_target = conformer.transform_raw(target_data['atoms'], target_data['coordinates'])

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = [self.data_prompt[index]['src_tokens'],
                      self.data_prompt[index]['src_distance'],
                      self.data_prompt[index]['src_edge_type'],
                      self.data_prompt[index]['src_coord'],
                      self.data_input[index]['src_tokens'],
                      self.data_input[index]['src_distance'],
                      self.data_input[index]['src_edge_type'],
                      self.data_input[index]['src_coord']]
        data_target = [self.data_target[index]['src_tokens'],
                       self.data_target[index]['src_distance'],
                       self.data_target[index]['src_edge_type'],
                       self.data_target[index]['src_coord']]
        return data_input, data_target


if __name__ == "__main__":
    text_path = os.path.join('../', 'source', 'mol.dict.txt')
    checkpoint_dir = '/home/junli/WORK/libin/model'
    dictionary_test = Dictionary.load(text_path)
    d_input, d_prompt, d_target = read_opt_files('./test')
    dataloader_train, dataloader_val = get_dataloader_active(d_input, d_prompt, d_target, dictionary_test, batch_size=4)
    print(len(dataloader_train))
    print(len(dataloader_val))
    model = get_model.load_model(text_path, checkpoint_dir)
    active_train(model, 1e-3, 3, 200, dataloader_train, dataloader_val, device='cpu', checkpoint_dir=checkpoint_dir, checkpoint_interval=5)

