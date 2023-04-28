import os
import numpy as np
import glob, yaml
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import loader
import barlow
from gnn import Action_GNN, Body_GNN


from torch.utils.tensorboard import SummaryWriter
import time
from time import strftime, localtime
from tqdm import tqdm


torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Pretrain(object):
    def __init__(self, args):
        with open('./code/config.yaml','r') as ymlfile:
            cfg = yaml.full_load(ymlfile) 

        self.seed = args.seed 
        self.root_dir = args.root_dir

        self.train_dir = os.path.join(self.root_dir, 'etri_train')

        self.device = args.device
        self.mode  = args.mode 

        self.save_root = args.save_root
        self.save_mode = args.save_mode
        self.exp_name = args.exp_name

        self.batch_size = args.batch_size 
        self.n_epoch = args.epoch
        self.spt_lr = args.spt_lr
        self.p = args.p  
        self.barlow_epoch = args.barlow_epoch
               
        self.train_loader = loader.dataloader(root_dir = self.train_dir,  batch_size = self.batch_size, mode = 'pretrain')

        self.action_edge = cfg['action_edge']
        self.action_edge = torch.tensor(self.action_edge, dtype=torch.long).t().contiguous().to(self.device)

        self.body_edge = cfg['body_edge']
        self.body_edge = torch.tensor(self.body_edge, dtype=torch.long).t().contiguous().to(self.device)
        
        self.action_nodes = cfg['action_nodes']
        self.action_kernel_size = cfg['action_kernel_size']
        self.action_gnn_outputdim = cfg['action_gnn_outputdim']

        self.body_nodes = cfg['body_nodes']
        self.body_kernel_size = cfg['body_kernel_size']
        self.body_gnn_outputdim = cfg['body_gnn_outputdim']
        self.classes = cfg['classes']

        self.action_spatio = Action_GNN(num_nodes = self.action_nodes, kernel_size = self.action_kernel_size, out_channels = self.action_gnn_outputdim, device=self.device).to(self.device)
        self.body_spatio = Body_GNN(num_nodes = self.body_nodes, kernel_size = self.body_kernel_size, out_channels = self.body_gnn_outputdim, device=self.device).to(self.device)
        self.linear = nn.Linear(self.action_gnn_outputdim + self.body_gnn_outputdim, self.classes).to(self.device)
        
        print('MODEL NETWORK')
        print('------------------ACTION SPATIO GNN------------------')
        print(self.action_spatio)
        print('------------------BODY SPATIO GNN------------------')
        print(self.body_spatio)
        print('------------------LINEAR MODEL------------------')
        print(self.linear)
        print('------------------------------------------------------')


        self.kappa = 0.93
        self.lbd_step = 100
        self.alpha = 0.99 
        lambd_init = torch.FloatTensor([0.0001])
        self.lambd = lambd_init.to(self.device)

        self.criterion  = nn.CrossEntropyLoss().to(self.device)
        self.barlow     = barlow.Barlow_loss().to(self.device)
        ### optimizer from COB paper.  
        self.optimizer  = torch.optim.Adam([{'params':self.action_spatio.parameters()}, 
                                            {'params':self.body_spatio.parameters()}, 
                                            {'params':self.linear.parameters()}
                                            ], lr = args.spt_lr, betas=(0.9, 0.98))   
                                             

    def train(self):
        print("Training START!")
        writer= SummaryWriter()
        self.body_spatio.train()
        self.action_spatio.train()
        self.linear.train()
        save_time = strftime('%Y-%m-%d %H:%M:%S', localtime(time.time()))
        n_total_steps = len(self.train_loader)
        for epoch in range(self.n_epoch):
            for iter, (x1, x2, target1) in enumerate((self.train_loader)):
                x1 = x1.type(torch.FloatTensor).squeeze(0).to(self.device) # body 
                x2 = x2.type(torch.FloatTensor).squeeze(0).to(self.device) # action 
                target = target1.type(torch.LongTensor).squeeze(0).to(self.device)
            
                self.optimizer.zero_grad()
                body_out, _ = self.body_spatio(x1, edge_index = self.body_edge) 
                action_out, _ = self.action_spatio(x2, edge_index = self.action_edge) 
                
                lin_in = torch.cat([body_out , action_out], 1)


                pred = self.linear(lin_in)
                loss_ce = self.criterion(pred, target)

                body_out2, _ = self.body_spatio(x1, edge_index = self.body_edge, drop_edge = True, p = self.p) 
                action_out2, _ = self.action_spatio(x2, edge_index = self.action_edge, drop_edge = True, p = self.p) 
                
                lin_in2 = torch.cat([body_out2 , action_out2], 1)
                on_diag, off_diag = self.barlow(lin_in, lin_in2) 
                loss_b = (on_diag + 0.005 * off_diag)
                if epoch>=self.barlow_epoch:
                    constraint = loss_b - self.kappa
                    loss = loss_ce + self.lambd * constraint
                else: 
                    loss = loss_ce

                target = target.squeeze(0)

                writer.add_scalar('conbarlow/con_loss' , loss_ce, epoch)
                if epoch>=self.barlow_epoch:
                    writer.add_scalar('conbarlow/barlow_loss', constraint, epoch)

                loss.backward()

                nn.utils.clip_grad_norm_(self.action_spatio.parameters(),3)
                nn.utils.clip_grad_norm_(self.body_spatio.parameters(),3)


                self.optimizer.step()

                with torch.no_grad():
                    if epoch>=self.barlow_epoch:
                        if epoch == 0 and iter == 0:
                            constraint_ma = constraint
                        elif epoch == self.barlow_epoch:
                            constraint_ma = constraint
                        else:
                            constraint_ma = self.alpha * constraint_ma.detach_() + (1 - self.alpha) * constraint
                        if iter % self.lbd_step == 0 :
                            self.lambd *= torch.clamp(torch.exp(constraint_ma), 0.9, 1.05)
                if (iter+1) % 5 == 0:
                    if epoch>=self.barlow_epoch:
                        print(f'epoch {epoch+1} / {self.n_epoch}, step {iter+1}/{n_total_steps}, total loss = {loss.item():.4f}, ssl_loss = {constraint.item():.4f}, ce_loss = {loss_ce:.4f}')
                    else:
                        print(f'epoch {epoch+1} / {self.n_epoch}, step {iter+1}/{n_total_steps}, total loss = {loss.item():.4f}, ce_loss = {loss_ce:.4f}')
           
            file_name = str(self.action_gnn_outputdim)+str(self.action_kernel_size)

            if (epoch+1) % 10== 0:
                if self.save_mode:   
                    if not os.path.exists(os.path.join(self.save_root, self.exp_name, file_name + '_' + save_time)):
                        os.makedirs(os.path.join(self.save_root, self.exp_name, file_name + '_' + save_time))
                        
                    torch.save(self.action_spatio.state_dict(), os.path.join(self.save_root, self.exp_name, file_name + '_' + save_time, 'action_spatio_{}epoch.pth'.format((epoch+1))))
                    torch.save(self.body_spatio.state_dict(), os.path.join(self.save_root, self.exp_name, file_name + '_' + save_time, 'body_spatio_{}epoch.pth'.format((epoch+1))))
                    torch.save(self.linear.state_dict(), os.path.join(self.save_root, self.exp_name, file_name + '_' + save_time, 'linear_{}epoch.pth'.format((epoch+1))))
