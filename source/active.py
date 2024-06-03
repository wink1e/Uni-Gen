import math
import torch
import torch.utils.data
import numpy as np
import time
import os
import get_data
import get_model
import test_model
from unicore.data import Dictionary
try:
    from unimol_tools.data.conformer import ConformerGen
except ModuleNotFoundError:
    from unimol.data.conformer import ConformerGen
# This script is the main program of active learning part.

text_path = os.path.join('../', 'source', 'mol.dict.txt')
dictionary_active = Dictionary.load(text_path)


def get_sys_info():
    env_dict = os.environ()
    if 'SLURM_JOB_NODELIST' in env_dict:
        host = str(env_dict['SLURM_NODELIST'])
        proc_num = int(env_dict['SLURM_NPROCS'])
        if_ssh = True
    else:
        raise "Currently only support slurm platform!"
    return proc_num, host, if_ssh


def run_cp2k(work_dir, cp2k_exe, parallel_exe, proc_num_per_node, proc_num, host):
    import subprocess
    host_info = ''.join(proc_num, '/', host)
    run_1 = f"""
#! /bin/bash
mpi_num={proc_num_per_node}
run_path={work_dir}
parallel_exe={parallel_exe}
mpi_num_arr=(${{mpi_num///}})
num=${{#mpi_num_arr[*]}}
for ((i=0;i<num;i++));
do
echo "$i $mpi_num_arr[i]"; done | $parallel_exe -j $num -S {host_info} --controlmaster --sshdelay 0.2 $run_path/produce.sh {{}} $run_path
"""
    produce_1 = f"""
#! /bin/bash
module load compilers/gcc/v9.3.0
source /apps/soft/cp2k/cp2k-9.1/tools/toolchain/install/setup
x=$1
run_path=$2
x_arr=(${{x///}})
sub_path=$run_path/sys_${{x_arr[0]}}
cd $sub_path
mpirun -np $x_arr[1] {cp2k_exe} $sub_path/cp2k.input 1> $sub_path/cp2k.out 2> $sub_path/cp2k.err
cd {work_dir}
"""
    run_file = os.path.join(work_dir, 'run.sh')
    produce_file = os.path.join(work_dir, 'produce.sh')
    with open(run_file, 'w') as f:
        f.write(run_1)
    with open(produce_file, 'w') as f:
        f.write(produce_1)
    subprocess.run('chmod +x run.sh', cwd=work_dir, shell=True)
    subprocess.run('chmod +x produce.sh', cwd=work_dir, shell=True)
    subprocess.run('bash ./run.sh', cwd=work_dir, shell=True)



def gen_one_frame(model, input_file: str, output_file: str) -> None:
    atoms, coord = test_model.xyz_to_list(input_file)
    input_dict = test_model.list_to_unimol_input_dict(atoms, coord, energies=[])
    input_dict = test_model.unimol_input_dict_to_unimol_model_dict(input_dict, text_path)
    input_loader = test_model.get_dataloader(input_dict, input_dict, batch_size=1, if_shuffle=False)
    output, _ = test_model.predict(model, input_loader, text_path, angle_range=(0, 90), distance=1.0, min_coord_num=3.0, device='cpu')
    atoms, coord = test_model.unimol_model_dict_to_list(output, text_path)
    test_model.list_to_xyz(output_file, atoms, coord)


def gen_cp2k_input(cp2k_input_file, output_dir, xyz_file_name):
    output_file_path = os.path.join(output_dir, "cp2k.input")
    with open(cp2k_input_file, 'r') as f:
        lines = f.readlines()
    output_file = open(output_file_path, 'w')
    for line in lines:
        if line.split()[0] == 'COORD_FILE_NAME':
            newline = line.split()
            newline[1] = xyz_file_name
            newline.insert(0, '            ')
            newline = '  '.join(newline)
            output_file.write(newline)
        else:
            output_file.write(line)


def active_learning_iter(old_checkpoint_path, cp2k_input, num_parallel, batch_size, criterion, max_iter):
    r"""
    this function is the main active learning iteration process.
    :param old_model:
    :param max_iter:
    :return:
    """
    iter_i = 0
    diff = float('inf')
    while iter_i < max_iter and diff > criterion:
        path_i = os.path.join('./', f'iter_{iter_i}')
        if not os.path.exists(path_i):
            os.makedirs(path_i)
        for i in range(num_parallel):
            sys_i = os.path.join(path_i, f'sys_{i}')
            os.makedirs(sys_i)
            # Initializing cp2k opt tasks
            gen_one_frame()
            gen_cp2k_input()
        run_cp2k()
        # Training process
        d_input, d_prompt, d_target = read_opt_files(path_i)
        train_loader, val_loader = get_dataloader_active(d_input, d_prompt, d_target, dictionary_active, batch_size=batch_size)
        if iter_i==0:
            model = get_model.load_model(text_path, old_checkpoint_path)
        else:
            model = get_model.load_model(text_path, os.path.join('./', f'iter_{iter_i-1}'))
        # Read criterion
        diff = active_train(model, 1e-3, 3, 200, train_loader, val_loader, device='cpu', checkpoint_dir=path_i, checkpoint_interval=50)
        

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
    begin_epoch_time = time.time()
    # training process
    model = old_model.to(device)
    for epoch in range(start_epoch, num_epoch):
        # training
        loss_epoch_train = 0
        loss_epoch_val = 0
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
            loss_epoch_train += loss.item()
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
                loss_epoch_val += loss.item()
                batch_time = time.time() - begin_time
                print(f"\r Batch {batch_i}/{num_batch}==Validation loss {loss.data:.8f}, Time {batch_time:.2f}s",
                      end='', flush=True)
            loss_epoch_val /= num_batch
            epoch_time = time.time() - begin_epoch_time
            print(f"\r Epoch {epoch}/{num_epoch}==Validation loss: {loss_epoch_val:.8f}, Time {epoch_time:.2f}", end='\t')
        scheduler.step(loss_epoch_val)
        loss_train_list.append(loss_epoch_train)
        loss_val_list.append(loss_epoch_val)

        # Save checkpoint file
        if (epoch+1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "loss_train_list": loss_train_list,
                          "loss_val_list": loss_val_list}
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}_epoch_active.pkl")
            torch.save(checkpoint, checkpoint_path)
    print(f"Training loss list: {loss_train_list}\n Validation loss list: {loss_val_list}")
    return loss_epoch_val


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
    active_train(model, 1e-3, 3, 200, dataloader_train, dataloader_val, device='cpu', checkpoint_dir=checkpoint_dir, checkpoint_interval=10)

