import argparse

from source import get_data

parser = argparse.ArgumentParser(description='''Dataset storage format: 
load_path
    |---dir_1
         |---1.ener
         |---1.xyz
    |---dir_2
         |---2.ener
         |---2.xyz
Each subfolder (dir_..) must provide files in xyz and ener (cp2k) formats.
This script will traverse the structures in each subfolder (dir_...) using the same slicing method, then aggregate and generate a pkl data file, which will be saved in load_path.
''')

parser.add_argument(
    "--load-path",
    type=str,
    metavar="P",
    help="file path to be loaded")

parser.add_argument(
    "--train-size",
    type=int,
    metavar="T",
    help="the data size to be used for training in every file.",
    default=50)

parser.add_argument(
    "--train-slice",
    type=int,
    metavar="S",
    help="the slice of the training set in data",
    default=10)

parser.add_argument(
    "--valid-size",
    type=int,
    metavar="T",
    help="the data size to be used for validation in every file.",
    default=10)

parser.add_argument(
    "--valid-slice",
    type=int,
    metavar="S",
    help="the slice of the validation set in data",
    default=50)

args = parser.parse_args()

load_path = args.load_path  
'''Dataset storage format: 
load_path
    |---dir_1
         |---1.ener
         |---1.xyz
    |---dir_2
         |---2.ener
         |---2.xyz
Each subfolder (dir_..) must provide files in xyz and ener (cp2k) formats.
'''
my_slice_train = slice(-args.train_slice * args.train_size, None, args.train_slice)

my_slice_val = slice(-args.valid_slice * args.valid_size + 1, None, args.valid_slice)

'''
This script will traverse the structures in each subfolder (dir_...) using the same slicing method, then aggregate and generate a pkl data file, which will be saved in load_path.
'''

get_data.preprocess_data_1(load_path)
get_data.preprocess_data_2(load_path=load_path, my_slice=my_slice_train, if_train=True, if_val=False)
get_data.preprocess_data_2(load_path=load_path, my_slice=my_slice_val, if_train=False, if_val=True)
# show how many frames
get_data.get_data_from_pkl(load_path=load_path, if_train=True, if_val=False)
get_data.get_data_from_pkl(load_path=load_path, if_train=False, if_val=True)
