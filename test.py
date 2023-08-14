import argparse

from source import test_model

parser = argparse.ArgumentParser(description='''
将一个或者多个已知的稳定结构xyz文件放在input_dir文件夹下
执行代码后将会在output_dir的子文件夹bad_dir和good_dir分别输出随机扰动后的xyz文件和模型输出的xyz文件
output_dir
 |---bad_dir
      |---bad_U2_0.xyz
      |---bad_U2_1.xyz
      |...
 |---good_dir
      |---good_U2_0.xyz
      |---good_U2_1.xyz
      |...
      
随机扰动的参数: 扰动角度angle_range范围越大, 扰动距离distance(单位:埃)越大,, 最低配位数min_coord_num越多, 扰动越剧烈.
''')

parser.add_argument(
    "--input-dir",
    type=str,
    metavar="I",
    help="input directory")
parser.add_argument(
    "--output-dir",
    type=str,
    metavar="O",
    help="output directory")
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    metavar="C",
    help="checkpoint directory")
parser.add_argument(
    "--start-angle",
    type=float,
    metavar="s",
    help="the start shift angle, should between (0, 90)",
    default=0.0)
parser.add_argument(
    "--end-angle",
    type=float,
    metavar="e",
    help="the end shift angle, should between (0, 90)",
    default=90.0)
parser.add_argument(
    "--distance",
    type=float,
    metavar="d",
    help="the shift maximum distance, default is 1.0 angstrom",
    default=1.0)
parser.add_argument(
    "--min-coord-num",
    type=int,
    metavar="n",
    help="minimum coordination that a atom will not shift, default is 3",
    default=5)
parser.add_argument(
    "--device",
    type=str,
    metavar="d",
    help="the device to be used for testing",
    choices=['cpu', 'cuda:0'],
    default='cpu')
args = parser.parse_args()

input_dir = args.input_dir  # 已知结构的存放位置
output_dir = args.output_dir  # 模型输出结构的存放位置
'''
将一个或者多个已知的稳定结构xyz文件放在input_dir文件夹下
执行代码后将会在output_dir的子文件夹bad_dir和good_dir分别输出随机扰动后的xyz文件和模型输出的xyz文件
output_dir
 |---bad_dir
      |---bad_U2_0.xyz
      |---bad_U2_1.xyz
      |...
 |---good_dir
      |---good_U2_0.xyz
      |---good_U2_1.xyz
      |...
'''
checkpoint_dir = args.checkpoint_dir  # 模型检查点的保存位置(本例中的检查点是训练400步后生成的)
'''
随机扰动的参数: 扰动角度angle_range范围越大, 扰动距离distance(单位:埃)越大,, 最低配位数min_coord_num越多, 扰动越剧烈.
'''
angle_range = (args.start_angle, args.end_angle)
distance = args.distance
min_coord_num = args.min_coord_num
device = args.device  # 模型的预测平台

test_model.test_model(input_dir=input_dir, output_dir=output_dir, checkpoint_dir=checkpoint_dir,
                      angle_range=angle_range, distance=distance, min_coord_num=min_coord_num, device=device)
