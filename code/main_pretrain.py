import argparse, os, torch, inspect
from pretrain import Pretrain
from util import str2bool
import pandas as pd


"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/home/dilab/data/etri")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--spt_lr', type=float, default=0.001)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--barlow_epoch', type=int, default=0) # barlow constraint epoch 
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--mode', type = str, default='conbarlow') 
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--exp_name', type=str, default='etri')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_mode', type=str2bool, default=True)
    parser.add_argument('--save_root', type=str, default="saved_models")


    return (parser.parse_args())


def main():
    args = parse_args()
    print(args)
    if args.save_root:
        if not os.path.join(args.save_root):
            os.mkdir(os.path.join(args.save_root))
    output = Pretrain(args)
    
    output.train()


if __name__ == '__main__':
    main()