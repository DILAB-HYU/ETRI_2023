from __future__ import print_function
import argparse
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from typing import Optional
from torch import Tensor
from torch_scatter import scatter

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            print(m)
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
    
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='mean')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def std_by_channel(data):

    # Calculate the mean and standard deviation of each channel
    channel_means = np.mean(data, axis=1, keepdims=True)
    channel_stds = np.std(data, axis=1, keepdims=True)
    zero_std_channels = np.where(channel_stds == 0)[0]
    channel_stds[zero_std_channels] = 1 

    # Normalize each channel using its mean and standard deviation
    normalized_data = (data - channel_means) / channel_stds

    return normalized_data



def detect_change(arr):
    change_points = [0]*len(arr)
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            change_points[i] = 1
    return change_points