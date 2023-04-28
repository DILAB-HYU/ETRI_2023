import os
import numpy as np
import glob, yaml
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import loader
from gnn import Action_GNN, Body_GNN
import barlow 
from tmse import _GaussianSimilarityTMSE
from torch.nn.utils import spectral_norm

class Pretrain(object):
    def __init__(self, args):
        with open('./code/config.yaml','r') as ymlfile:
            cfg = yaml.full_load(ymlfile) 

        self.seed = args.seed 
        self.root_dir = args.root_dir
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.test_dir = os.path.join(self.root_dir, 'test')
        
        self.save_file_name = args.save_file_path.split("/")[-1] # segmentation file 이 저장될 버전 이름 (e.g., conbarlow_gin_5_epcoh)

        self.body_spatio_file = os.path.join(args.save_file_path, args.body_spatio_file) 
        self.action_spatio_file = os.path.join(args.save_file_path, args.action_spatio_file)
        self.linear_file =os.path.join(args.save_file_path, args.linear_file) 
        self.device = args.device
        self.mode  = args.mode 
        self.lambda_g = args.lambda_g
        self.save_root = args.save_root
        self.save_mode = args.save_mode
        self.csv_save = args.csv_save
        self.report_name = args.report_name
        self.exp_name = args.exp_name

        self.batch_size = args.batch_size 
        self.n_epoch = args.epoch
        self.spt_lr = args.spt_lr
        self.p = args.p  
        self.lambda_b = 0.1
        self.barlow_epoch = args.barlow_epoch
        self.pretrain_epoch = args.pretrain_epoch 
               
        self.train_loader = loader.dataloader(root_dir = self.train_dir,  batch_size = 1, mode = 'segmentation')
        self.test_loader = loader.dataloader(root_dir = self.test_dir, batch_size = 1, mode = 'segmentation')
        print(self.test_loader.dataset.__len__())

        ##############
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
        
        self.gaussian_tmse = _GaussianSimilarityTMSE()

        ###############
        self.action_spatio.load_state_dict(torch.load(self.action_spatio_file), strict=False)
        self.body_spatio.load_state_dict(torch.load(self.body_spatio_file), strict=False)
        self.linear.load_state_dict(torch.load(self.linear_file))
        self.criterion = nn.CrossEntropyLoss().to(self.device)
 
        self.optimizer  = torch.optim.Adam([{'params':self.linear.parameters()}], 
                                            lr = args.spt_lr, betas=(0.9, 0.98))

        self.layernorm = nn.LayerNorm(128).to(self.device)
        self.linear = spectral_norm(self.linear)

    def train(self):
        print("Training START!")
       
        self.body_spatio.eval()
        self.action_spatio.eval()
        self.linear.train() 
        loss = 0.0
        n_total_steps = len(self.train_loader)
        for epoch in range(self.n_epoch):
            for iter, (x1, x2, target1, target2) in enumerate(self.train_loader):
                x1 = x1.type(torch.FloatTensor).squeeze(0).to(self.device) # body 
                x2 = x2.type(torch.FloatTensor).squeeze(0).to(self.device) # action 
                target = target1.type(torch.LongTensor).squeeze(0).to(self.device)
                target2 = torch.LongTensor(target2).squeeze(0).to(self.device)
                batch_loader = loader.batchloader(x1, x2, target, target2, batch_size = self.batch_size, shuffle=False, drop_last = False)
                print("x1:{}, x2:{}, target:{}".format(x1.shape, x2.shape, target.shape))
                self.optimizer.zero_grad()

                for (b_x, b_x2, b_target, b_target2) in batch_loader: #
                    
                    if b_target.shape[0] > 1:
                        body_out, body_pred = self.body_spatio(b_x, edge_index = self.body_edge) 
                        action_out, aciton_pred = self.action_spatio(b_x2, edge_index = self.action_edge) 
                        
                        lin_in = torch.cat([body_out , action_out], 1)

                        pred = self.linear(lin_in)
                        pred = pred.squeeze(0)
                        
                        b_target = b_target.squeeze(0)
                        loss1 = self.criterion(pred, b_target)
                        loss2 = self.gaussian_tmse(preds = pred, gts = b_target, sim_index = lin_in)
                        print("ce loss:{}, gtmse loss:{}".format(loss1, loss2))
                        loss = loss1 + self.lambda_g*loss2
                        loss.backward()

                        self.optimizer.step()


                if (iter+1) % 1 == 0:
                    print(f'epoch {epoch+1} / {self.n_epoch}, step {iter+1}/{n_total_steps}, loss = {loss.item():.4f}')
            

            file_name = str(self.action_gnn_outputdim)+str(self.action_kernel_size)
            
            if (epoch+1) % 5 == 0:
                if self.save_mode:   
                    if not os.path.exists(os.path.join(self.save_root, self.exp_name, file_name)):
                        os.makedirs(os.path.join(self.save_root, self.exp_name, file_name))
                        
                    torch.save(self.action_spatio.state_dict(), os.path.join(self.save_root, 'seg_action_spatio_{}_{}epoch.pth'.format(self.save_file_name, (epoch+1))))
                    torch.save(self.body_spatio.state_dict(), os.path.join(self.save_root,'seg_body_spatio_{}_{}epoch.pth'.format(self.save_file_namee,(epoch+1))))
                    torch.save(self.linear.state_dict(), os.path.join(self.save_root,'seg_linear_spatio_{}_{}epoch.pth'.format(self.save_file_name,(epoch+1))))
            

    def test(self):
        print('Final Prediction Start')
        with torch.no_grad():
            self.body_spatio.eval()
            self.action_spatio.eval()

            self.linear.eval()

            predlist=torch.zeros(0, dtype=torch.long, device=self.device)
            lbllist=torch.zeros(0, dtype=torch.long, device=self.device)

            for _, (x1, x2, target, target2) in enumerate(self.test_loader): 
                self.optimizer.zero_grad()
                x1 = x1.type(torch.FloatTensor).squeeze(0).to(self.device)
                x2 = x2.type(torch.FloatTensor).squeeze(0).to(self.device)
                target = target.type(torch.LongTensor).squeeze(0).to(self.device)  
                target2 = torch.LongTensor(target2).squeeze(0).to(self.device)

                self.optimizer.zero_grad()
                batch_loader = loader.batchloader(x1, x2, target, target2, batch_size = self.batch_size, shuffle=False, drop_last = False)               
                    
                for (b_x, b_x2, b_target, b_target2) in batch_loader:
                    body_out, body_pred = self.body_spatio(b_x, edge_index = self.body_edge) 
                    action_out, aciton_pred = self.action_spatio(b_x2, edge_index = self.action_edge) 

                    lin_in = torch.cat([body_out , action_out], 1)
                    pred = self.linear(lin_in)

                    _, predictions = torch.max(pred,-1)

                    predlist = torch.cat([predlist, predictions.view(-1)])
                    lbllist = torch.cat([lbllist, b_target.view(-1)])

            predlist, lbllist = predlist.to('cpu'), lbllist.to('cpu')
            acc = accuracy_score(lbllist, predlist)
            f1 = f1_score( lbllist, predlist, average='weighted')
            class_f1 = f1_score( lbllist, predlist, average=None)

            report = classification_report(lbllist, predlist, output_dict = False)
            print(report)
            print(confusion_matrix( lbllist, predlist))
            
            self.acc = acc
            self.f1 = f1
            self.class_f1 = class_f1
            if self.csv_save:
                self.save_report(self)
        
        return acc, f1, class_f1
    
    def save_report(self, result_only = False):
        
        ### save result with all params 
        df_col = ['mode', 
                  'exp_name', 
                  'epoch', 

                  'body_gnn_outputdim',
                  'action_gnn_outputdim', 

                  'action_kernel_size', 
                  'body_kernel_size', 

                  'pretrain_lr', 
                  'batch_size', 
                  'acc', 'f1', 'class_f1']


        data = {'mode':self.mode,
                'exp_name':self.exp_name, 
                'epoch' : self.n_epoch, 
                'body_gnn_outputdim': self.body_gnn_outputdim, 
                'action_gnn_outputdim': self.action_gnn_outputdim, 

                'action_kernel_size': self.action_kernel_size,
                'body_kernel_size': self.body_kernel_size,

                'pretrain_lr': self.spt_lr, 
                'batch_size': self.batch_size, 
                'acc':self.acc, 'f1': self.f1, 
                'class_f1' : self.class_f1
                }

        df = pd.DataFrame.from_records([data])
        df.to_csv(self.report_name, mode='a', header=False, index=False)

        if result_only:
            ##### save result only 
            result_path = os.path.join(self.report_name[:-4] +"_result.csv")
            df_col = ['mode', 'exp_name','seed', 'batch_size', 'acc', 'f1'] #, 'class f1']
            if not os.path.exists(result_path): 
                df = pd.DataFrame([], columns=df_col)
                df.to_csv(result_path, header=True, index=False)

            data = {'mode':self.mode, 'exp_name': self.exp_name, 'seed': self.seed,'batch_size': self.batch_size, 'acc': self.acc, 'f1': self.f1}
            df = pd.DataFrame.from_records([data])
            df.to_csv(result_path, mode='a', header=False, index=False)