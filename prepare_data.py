import argparse

from source import get_data

parser = argparse.ArgumentParser(description='''数据集的存放格式: 
load_path
    |---dir_1
         |---1.ener
         |---1.xyz
    |---dir_2
         |---2.ener
         |===2.xyz
每个子文件夹(dir_..)下, 需要提供xyz和ener(cp2k)格式的文件
本代码将遍历每一个子文件夹（dir_...）中的结构采取相同的切片方式，然后汇总生成pkl数据文件，并且保存在load_path
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

load_path = args.load_path  # 原始数据xyz文件和ener文件的保存路径, 也是训练集pkl文件的保存路径
'''数据集的存放格式: 
load_path
    |---dir_1
         |---1.ener
         |---1.xyz
    |---dir_2
         |---2.ener
         |===2.xyz
每个子文件夹(dir_..)下, 需要提供xyz和ener(cp2k)格式的文件
'''
my_slice_train = slice(-args.train_slice * args.train_size, None, args.train_slice)
# 从load_path的每个文件夹下读取xyz和ener文件， 在最后500个结构中每隔10个数据提取一个结构, 保存为pkl文件，用于训练
my_slice_val = slice(-args.valid_slice * args.valid_size + 1, None, args.valid_slice)
# 从load_path的每个文件夹下读取xyz和ener文件， 在最后499个结构中每隔50个数据提取一个结构, 保存为pkl文件，用于验证
'''
本代码将遍历每一个子文件夹（dir_...）中的结构采取相同的切片方式，然后汇总生成pkl数据文件，并且保存在load_path
'''

get_data.preprocess_data_1(load_path)
get_data.preprocess_data_2(load_path=load_path, my_slice=my_slice_train, if_train=True, if_val=False)
get_data.preprocess_data_2(load_path=load_path, my_slice=my_slice_val, if_train=False, if_val=True)
# show how many frames
get_data.get_data_from_pkl(load_path=load_path, if_train=True, if_val=False)
get_data.get_data_from_pkl(load_path=load_path, if_train=False, if_val=True)
