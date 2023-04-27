import argparse, os, torch, inspect
from segmentation_v2 import Pretrain
from util import str2bool
import pandas as pd


"""parsing and configuration"""
def parse_args():


    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/home/dilab/data/etri/daywise") #/home/dilab/data/etri/daywise   #/home/dilab/data/userwise/user1
    
    parser.add_argument('--save_file_path', type=str, default="./share/conbarlow_gin") 
    parser.add_argument('--body_spatio_file', type=str, default="body_spatio_50epoch.pth") 
    parser.add_argument('--action_spatio_file', type=str, default="action_spatio_50epoch.pth")
    parser.add_argument('--linear_file', type=str, default="linear_50epoch.pth") 

    parser.add_argument('--batch_size', type=int, default=128) # 1024 is better
    parser.add_argument('--spt_lr', type=float, default=0.001)
    parser.add_argument('--lambda_g', type=float, default=1)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--barlow_epoch', type=int, default=1) # barlow constraint epoch 
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--mode', type = str, default='segmentation') ## 'supervised' , 'supconbarlow', 'supbarlow', 'conbarlow'
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--exp_name', type=str, default='etri')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--report_name', type=str, default='report.csv')
    parser.add_argument('--csv_save', type=str2bool, default=True)
    parser.add_argument('--save_mode', type=str2bool, default=False)
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

    # if args.mode == 'supervised':
    acc, f1, class_f1 = output.test()

if __name__ == '__main__':
    main()