import torch
import torch.nn as nn
import numpy as np

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Barlow_loss(nn.Module):
    def __init__(self, out_dim = 128, device = 'cpu'):
        super(Barlow_loss, self).__init__()
        self.device = device
        #self.batchsize = batchsize
        self.bn = nn.BatchNorm1d(out_dim, affine=False).to(self.device) # normalize along batch dim
        
        self.fc = nn.Sequential(nn.Linear(128, 128), 
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(0.2)).to(self.device) 

    def forward(self, z1, z2, off_neg = False):
        # z1(batch, z2)
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2) # matrix multiplication

        # sum the cross-correlation matrix between all gpus
        # inplace version of torch.div(input, other): out = input/other
        c.div_(z1.shape[0]) 

        #torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        # https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/main.py
        #off_diag = off_diagonal(c).add_(1).pow_(2).sum()
        
        return on_diag, off_diag