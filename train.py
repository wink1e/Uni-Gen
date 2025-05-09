import argparse

from source import train_model

parser = argparse.ArgumentParser(description=
                                 """
                                 Training the model.
                                 Please run prepare_data first for the data generation.
                                 """)

parser.add_argument(
    "--lr",
    type=float,
    metavar="l",
    help="the learning rate of the training process",
    default=1e-3)
parser.add_argument(
    "--lr-patience",
    type=int,
    metavar="p",
    help="the patience for the decrease of the learning rate."
         " the smaller the patience, the faster the decrease",
    default=3)
parser.add_argument(
    "--num-epoch",
    type=int,
    metavar="e",
    help="the epoch for the training",
    default=10)
parser.add_argument(
    "--batch-size",
    type=int,
    metavar="b",
    help="the batch size of the training process, the exponential of two is suggested.",
    default=4)
parser.add_argument(
    "--pkl_dir",
    type=str,
    metavar="p",
    help="the training data which is stored in .pkl file, please check the argument in prepare_data.py")
parser.add_argument(
    "--device",
    type=str,
    metavar="d",
    help="the device to be used for training and validation",
    choices=['cpu', 'cuda:0'],
    default='cpu')
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    metavar="C",
    help="the directory to store checkpoint file.")
parser.add_argument(
    "--checkpoint-interval",
    type=int,
    metavar="i",
    help="the interval of the checkpoint saving.",
    default=5)
parser.add_argument(
    "--if-read-checkpoint",
    type=bool,
    metavar="r",
    help="whether or not reading the previous checkpoint file.",
    default=False)

args = parser.parse_args()

lr = args.lr  
lr_patience = args.lr_patience  
num_epoch = args.num_epoch  
batch_size = args.batch_size  
pkl_dir = args.pkl_dir  
device = args.device  

checkpoint_dir = args.checkpoint_dir  
checkpoint_interval = args.heckpoint_interval  
if_read_checkpoint = args.if_read_checkpoint

train_model.train_model(lr=lr, lr_patience=lr_patience, num_epoch=num_epoch, batch_size=batch_size,
                        pkl_dir=pkl_dir, device=device,
                        checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
                        if_read_checkpoint=if_read_checkpoint)
