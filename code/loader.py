import os
import numpy as np
import glob
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
from util import std_by_channel, detect_change

class Data(Dataset):
    def __init__(self, root_dir, sensor_1 = ['e4Bvp', 'e4Eda', 'e4Hr', 'e4Temp', 'mGps_lat', 'mGps_lon', 'mGps_acc'],
                                 sensor_2 = ['e4Acc_x', 'e4Acc_y', 'e4Acc_z', 'mAcc_x', 'mAcc_y', 'mAcc_z', 
                                             'mGyr_x', 'mGyr_y', 'mGyr_z', 'mGyr_roll', 'mGyr_pitch', 'mGyr_yaw', 
                                             'mMag_x', 'mMag_y', 'mMag_z'], normalization=True, mode = 'pretrain'
                ):
        '''
        Class Data : Dataset Class for sensor data. 
        
        Params:
        - root_dir : data_dir
        - sensor_1 : sr 240인 센서들
        - sensor_2 : sr 1920인 센서들
        
        Returns:
        - x1: sr 240인 센서들 채널별로 스택 [센서수, 갯수, 240]
        - x2: sr 1920인 센서들 채널별로 스택 [센서수, 갯수, 1920인]
        - targets1: One-hot encoded action labels
        - targets2: One-hot encoded actionOption labels
        
        '''

        # data path  
        self.root_dir = root_dir 
        self.sensor_1 = sensor_1 # 생리반응
        self.sensor_2 = sensor_2  # 신체움직임 
        self.normalization = normalization
        self.file_names = glob.glob(os.path.join(root_dir,'*.npz'))
        self.le = preprocessing.LabelEncoder()
        self.mode = mode

    def __len__(self):
        return len(self.file_names)
                
    def __getitem__(self, idx):

        file = np.load(self.file_names[idx])
        f_name = self.file_names[idx]

        x1 = []
        x1 = ([file[ch] for ch in self.sensor_1])
        x1 = np.stack(x1, axis=0)
        if self.normalization:
            x1 = std_by_channel(x1)
        x1 = torch.from_numpy(x1)


        x2 = []
        x2 = ([file[ch] for ch in self.sensor_2])
        x2 = np.stack(x2, axis=0)
        if self.normalization:
            x2 = std_by_channel(x2)
        x2 = torch.from_numpy(x2)

         # 3000만 normalize 되도록 다시 구현
        
        action_label = file['label_action']
        action_option_label = file['label_actionOption']

        if self.mode == 'pretrain':
            return x1, x2, action_label
        elif self.mode == 'segmentation':
            boundary_label = detect_change(action_label)
            return x1.permute(1,0,2), x2.permute(1,0,2), action_label.reshape(-1), boundary_label

class Batch_Dataset(Dataset):
    def __init__(self, x, x2, y, y2):
        super(Batch_Dataset, self).__init__()

        self.x = x
        self.x2 = x2
        self.y = y
        self.y2 = y2
        self.len = x.shape[0]


    def __getitem__(self, index):
        try:
            return self.x[index], self.x2[index], self.y[index], self.y2[index]
        except IndexError:
            print('indexerror')
            return self.x.squeeze(0), self.x2.squeeze(0), self.y.squeeze(0), self.y2.squeeze(0)
    def __len__(self):
        return self.len

def dataloader(root_dir, batch_size, mode):
    dataset = Data(root_dir = root_dir, sensor_1 = ['e4Bvp', 'e4Eda', 'e4Hr', 'e4Temp'],
                                        sensor_2 = ['e4Acc_x', 'e4Acc_y', 'e4Acc_z', 'mAcc_x', 'mAcc_y', 'mAcc_z', 
                                             'mGyr_x', 'mGyr_y', 'mGyr_z', 'mGyr_roll', 'mGyr_pitch', 'mGyr_yaw', 
                                             'mMag_x', 'mMag_y', 'mMag_z'], normalization=True, mode = mode)
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)
    return data_loader    

def batchloader(x, x2, y, y2,batch_size, shuffle, drop_last):
    dataset = Batch_Dataset(x, x2, y, y2)
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=shuffle, drop_last = drop_last)
    return data_loader