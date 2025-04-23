import argparse

from source import test_model

parser = argparse.ArgumentParser(description='''
Place one or more known stable structure xyz files in the input_dir folder.
After executing the code, the subfolders bad_dir and good_dir in the output_dir will respectively output the randomly perturbed xyz files and the xyz files predicted by the model.
output_dir
 |---bad_dir
    |---bad_U2_0.xyz
    |---bad_U2_1.xyz
    |...
 |---good_dir
    |---good_U2_0.xyz
    |---good_U2_1.xyz
    |...
    
Parameters for random perturbation: The larger the angle_range of the perturbation angle, the greater the perturbation distance (unit: angstrom), and the higher the minimum coordination number min_coord_num, the more intense the perturbation.
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
    help="the start shift angle, should be between (0, 90)",
    default=0.0)
parser.add_argument(
    "--end-angle",
    type=float,
    metavar="e",
    help="the end shift angle, should be between (0, 90)",
    default=90.0)
parser.add_argument(
    "--distance",
    type=float,
    metavar="d",
    help="the maximum shift distance, default is 1.0 angstrom",
    default=1.0)
parser.add_argument(
    "--min-coord-num",
    type=int,
    metavar="n",
    help="minimum coordination number for an atom to not shift, default is 3",
    default=5)
parser.add_argument(
    "--device",
    type=str,
    metavar="d",
    help="the device to be used for testing",
    choices=['cpu', 'cuda:0'],
    default='cpu')
args = parser.parse_args()

input_dir = args.input_dir  
output_dir = args.output_dir  
'''
Place one or more known stable structure xyz files in the input_dir folder.
After executing the code, the subfolders bad_dir and good_dir in the output_dir will respectively output the randomly perturbed xyz files and the xyz files predicted by the model.
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
checkpoint_dir = args.checkpoint_dir  
'''
Parameters for random perturbation: The larger the angle_range of the perturbation angle, the greater the perturbation distance (unit: angstrom), and the higher the minimum coordination number min_coord_num, the more intense the perturbation.
'''
angle_range = (args.start_angle, args.end_angle)
distance = args.distance
min_coord_num = args.min_coord_num
device = args.device  

test_model.test_model(input_dir=input_dir, output_dir=output_dir, checkpoint_dir=checkpoint_dir,
                angle_range=angle_range, distance=distance, min_coord_num=min_coord_num, device=device)
