import pandas as pd
import numpy as np
import os
import tqdm
import pickle


class CP2K_MD:
    r"""
    The class to read from CP2K traj. file.
    """

    def __init__(self, path):
        self.path = path
        self.time = []
        self.pot, self.pot_origin = [], []
        self.temp = []
        self.xx, self.xy, self.xz, \
            self.yx, self.yy, self.yz, \
            self.zx, self.zy, self.zz \
            = [], [], [], [], [], [], [], [], []
        self.x, self.y, self.z = [], [], []
        self.box = []
        self.vol = []
        self.coord = []
        self.atom_type = []
        self.force = []
        self.velocity = []

    def load_traj_atom_type(self):
        r"""self.atom_type: numpy array, shape=(natoms,)
        """
        file_path = None
        for file in os.listdir(self.path):
            if file.endswith('.xyz') and ('pos-1' in file):
                file_path = os.path.join(self.path, file)
                break
        assert file_path is not None
        with open(file_path, 'r') as f:
            atom_types_list = []
            num_atoms = int(f.readline())
            comment = f.readline()
            for i in range(num_atoms):
                line = f.readline().split()
                atom_types_list.append(str(line[0]))
            self.atom_type = np.array(atom_types_list)

    def load_traj_pot(self):
        r"""self.pot: numpy array, shape=(nframe,)"""
        file_path = None
        try:
            for file in os.listdir(self.path):
                if file.endswith('.ener'):
                    file_path = os.path.join(self.path, file)
                    break
            assert file_path is not None
            data = pd.read_table(file_path, engine='python', sep='\s+', header=None, comment='#')
            self.time = np.array(data.loc[:, 1])
            self.pot_origin = np.array(data.loc[0, 4])
            self.pot = np.array(data.loc[:, 4])
        except AssertionError:
            for file in os.listdir(self.path):
                if file.endswith('.xyz') and ('pos-1' in file):
                    file_path = os.path.join(self.path, file)
                    break
            assert file_path is not None
            with open(file_path, 'r') as f:
                pot = []
                n_atoms = int(f.readline())
                lines = f.readlines()
                lines = lines[::(n_atoms+2)]
                for line in lines:
                    pot.append(float(line.split()[5]))
            self.pot = pot


    def load_traj_coord(self):
        r"""self.coord: numpy array, shape=(nframe, 3*natoms)"""
        file_path = None
        for file in os.listdir(self.path):
            if file.endswith('.xyz') and ('pos-1' in file):
                file_path = os.path.join(self.path, file)
                break
        assert file_path is not None
        with open(file_path, 'r') as f:
            frames = []
            while True:
                try:
                    num_atoms = int(f.readline())
                    comment = f.readline()
                    positions = np.zeros((num_atoms, 3))
                    for i in range(num_atoms):
                        line = f.readline().split()
                        positions[i] = [float(x) for x in line[1:4]]
                    positions = np.reshape(positions, num_atoms * 3)
                    frames.append(positions)
                except:
                    break
            self.coord = np.array(frames)

    def load_traj_all(self):
        print(f'Loading from {self.path}')
        self.load_traj_coord()
        self.load_traj_atom_type()
        self.load_traj_pot()
        assert len(self.atom_type) * 3 == self.coord.shape[1]
        assert len(self.coord) == len(self.pot), f'len(self.coord)={len(self.coord)}\tlen(self.pot)={len(self.pot)}'
        print('Loading finished!')

    def save_traj_all(self, save_dir_path):
        path = save_dir_path
        os.makedirs(path, exist_ok=True)
        print(f'Saving trajectory into {path}')
        np.savetxt(os.path.join(path, 'atom_type'), self.atom_type, fmt="%s")
        np.save(os.path.join(path, 'coord'), self.coord)
        np.save(os.path.join(path, 'pot'), self.pot)
        print('Saving finished!')

    def clean_traj_all(self):
        self.atom_type = []
        self.coord = []
        self.pot = []

    def load_traj_all_quick(self, load_dir_path):
        self.clean_traj_all()
        assert os.path.exists(load_dir_path), f"load_dir_path '{load_dir_path}' does not exist"
        path = load_dir_path
        print(f'Loading trajectory from {path}')
        with open(os.path.join(path, 'atom_type'), 'r') as f:
            atom_type = [str(atom_type_i.split()[0]) for atom_type_i in f.readlines() if str(atom_type_i) != '\n']
            self.atom_type = np.array(atom_type)
        self.coord = np.load(os.path.join(path, 'coord.npy'))
        self.pot = np.load(os.path.join(path, 'pot.npy'))
        print('Loading finished!!')


def define_data(load_path: str):
    r"""
    coordinates should be List of numpy array whose shape=(natoms, 3)
    """
    print('Warning: function define_data in get_data.py is from old_version!')
    cp2k_data = CP2K_MD(path=load_path)
    cp2k_data.load_traj_all_quick(load_dir_path=load_path)
    target = cp2k_data.pot[::10]
    nframes = len(target)
    atoms = [cp2k_data.atom_type for _ in range(nframes)]
    coordinates = list(cp2k_data.coord[::10].reshape(nframes, -1, 3))
    custom_data = {'target': target,
                   'atoms': atoms,
                   'coordinates': coordinates,
                   }
    return custom_data


def preprocess_data_1(load_path: str) -> None:
    r"""
    You need to put all xyz and ener files into relevent directory in load_path.
    pot.npy, atom_type(txt file), coord.npy will be generated and saved.
    """
    assert os.path.exists(load_path)
    for i, subdir in enumerate(os.listdir(load_path)):
        subdir = os.path.join(load_path, subdir)
        if os.path.isdir(subdir):
            print(f'preprocess_data_1 {i}/{len(os.listdir(load_path))}:')
            cp2k_data = CP2K_MD(path=subdir)
            cp2k_data.load_traj_all()
            cp2k_data.save_traj_all(save_dir_path=subdir)
    print('Run preprocess_data_1 Successfully!')


def preprocess_data_2(load_path: str, my_slice: slice, if_train: bool, if_val: bool) -> None:
    r"""
    Load pot.npy, atom_type(txt file), coord.npy from all directory in load_path/,
    Transform them into Uni-Mol's data type,
    Save as pkl file in (if_train=True)load_path/data_target_train.pkl and load_path/data_prompt_train.pkl Or
        (if_train=False, if_val=True)load_path/data_target_validation.pkl and load_path/data_prompt_validation.pkl Or
        (if_train=False, if_val=False)load_path/data_target_test.pkl and load_path/data_prompt_test.pkl

    my_slice: Each trajectory information including atom type, coordinates and energy can be seen as
        A List(len=nframe), my_slice is used to slice the List of Each trajectory.
            Notice there are many trajectories, so this function will slice ntrajectory times,
        and concatenate them into pkl file.

    What is Uni-Mol's data type? Dict
    Each Dict contains 'target', 'atoms', 'coordinates'
    Dict['target']: energy, 1d-List of float, len=(nframe)
    Dict['atoms']: atom type, 2d-List string, len=(nframe, natom)
    Dict['coordinates']: coordinates, 3d-List of float, len=(nframe, natom, 3)
    """
    assert os.path.exists(load_path)
    data_target = {'atoms': [], 'coordinates': [], 'target': []}
    data_prompt = {'atoms': [], 'coordinates': [], 'target': []}

    for subdir in tqdm.tqdm(os.listdir(load_path)):
        subdir = os.path.join(load_path, subdir)
        if os.path.isdir(subdir):
            cp2k_data = CP2K_MD(path=subdir)
            cp2k_data.load_traj_all_quick(load_dir_path=subdir)
            nframes = len(cp2k_data.pot[my_slice])
            data_target['atoms'] += [cp2k_data.atom_type for _ in range(nframes)]
            data_target['coordinates'] += list(cp2k_data.coord[my_slice].reshape(nframes, -1, 3))
            data_target['target'] += list(cp2k_data.pot[my_slice])
            prompt_id = np.argmin(cp2k_data.pot)
            data_prompt['atoms'] += [cp2k_data.atom_type for _ in range(nframes)]
            data_prompt['coordinates'] += [cp2k_data.coord[prompt_id].reshape(-1, 3) for _ in range(nframes)]
            data_prompt['target'] += list([cp2k_data.pot[prompt_id] for _ in range(nframes)])

    assert if_train is False or if_val is False, 'Both if_train and if_val are True! '
    if if_train:
        data_target_path = os.path.join(load_path, 'data_target_train.pkl')
        data_prompt_path = os.path.join(load_path, 'data_prompt_train.pkl')
    elif if_val:
        data_target_path = os.path.join(load_path, 'data_target_validation.pkl')
        data_prompt_path = os.path.join(load_path, 'data_prompt_validation.pkl')
    else:
        data_target_path = os.path.join(load_path, 'data_target_test.pkl')
        data_prompt_path = os.path.join(load_path, 'data_prompt_test.pkl')

    with open(data_target_path, 'wb') as f:
        pickle.dump(data_target, f)
    with open(data_prompt_path, 'wb') as f:
        pickle.dump(data_prompt, f)
    print(f'Generate {data_target_path}\n And {data_prompt_path} Successfully!')


def get_data_from_pkl(load_path: str, if_train: bool, if_val: bool):
    r"""
    Load train data from load_path/data_target_train.pkl and load_path/data_prompt_train.pkl
    Or load validation data from load_path/data_target_validation.pkl and load_path/data_prompt_validation.pkl
    Or load test data from load_path/data_target_test.pkl and load_path/data_prompt_test.pkl

    And return TWO Dict tuple (data_target Dict, data_prompt Dict)
    Each Dict contains 'target', 'atoms', 'coordinates'
    Dict['target']: energy, 1d-List of float, len=(nframe)
    Dict['atoms']: atom type, 2d-List string, len=(nframe, natom)
    Dict['coordinates']: coordinates, 3d-List of float, len=(nframe, natom, 3)
    """
    assert if_train is False or if_val is False, 'Both if_train and if_val are True! '
    if if_train:
        data_target_path = os.path.join(load_path, 'data_target_train.pkl')
        data_prompt_path = os.path.join(load_path, 'data_prompt_train.pkl')
    elif if_val:
        data_target_path = os.path.join(load_path, 'data_target_validation.pkl')
        data_prompt_path = os.path.join(load_path, 'data_prompt_validation.pkl')
    else:
        data_target_path = os.path.join(load_path, 'data_target_test.pkl')
        data_prompt_path = os.path.join(load_path, 'data_prompt_test.pkl')

    with open(data_target_path, 'rb') as f:
        data_target_dict = pickle.load(f)
    with open(data_prompt_path, 'rb') as f:
        data_prompt_dict = pickle.load(f)
    if if_train:
        print(f"Train data: {len(data_target_dict['atoms'])} frames")
    elif if_val:
        print(f"Validation data: {len(data_target_dict['atoms'])} frames")
    else:
        print(f"Test data: {len(data_target_dict['atoms'])} frames")

    return data_target_dict, data_prompt_dict


if __name__ == '__main__':
    load_path = '../ucluster_data_0731'
    # preprocess_data_1(load_path) just need once
    preprocess_data_2(load_path=load_path, my_slice=slice(-500, None, 2), if_train=True, if_val=False)
    preprocess_data_2(load_path=load_path, my_slice=slice(-499, None, 10), if_train=False, if_val=True)
    # preprocess_data_2(load_path=load_path, my_slice=slice(-499, None, 5), if_train=False, if_val=False)
    # See how many frames
    get_data_from_pkl(load_path=load_path, if_train=True, if_val=False)
    get_data_from_pkl(load_path=load_path, if_train=False, if_val=True)
    # get_data_from_pkl(load_path=load_path, if_train=False, if_val=False)
