# function
import os
import numpy as np
import pandas as pd
import natsort
import argparse
from util import pre_e4Acc, pre_e4Bvp, pre_e4Eda, pre_e4Hr, pre_e4Temp, pre_mAcc, pre_mGps, pre_mGyr, pre_mMag, pre_label


def save_daywise_file(user_path, days, save_dir):

    # print(user_path)
    # print(user_path.split('/'))
    label = pd.read_csv(os.path.join(user_path, days, days +'_label.csv'))
    label_action = label['action']
    label_action = pre_label(label_action)
    label_actionOption = label['actionOption']
    ts = label['ts']

    e4Acc_x_,e4Acc_y_,e4Acc_z_,e4Bvp_,e4Eda_,e4Hr_,e4Temp_,mAcc_x_, mAcc_y_, mAcc_z_,mGps_lat_, mGps_lon_, mGps_acc_,mGyr_x_, mGyr_y_, mGyr_z_,mGyr_roll_, mGyr_pitch_, mGyr_yaw_, mMag_x_, mMag_y_, mMag_z_= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    ts_, label_action_, label_actionOption_ = [], [],[]

    for idx in range(len(ts)):
        e4Acc_path = os.path.join(user_path, days, 'e4Acc', str(int(ts[idx])))+'.csv'
        e4Bvp_path = os.path.join(user_path, days, 'e4Bvp', str(int(ts[idx])))+'.csv'
        e4Eda_path = os.path.join(user_path, days, 'e4Eda', str(int(ts[idx])))+'.csv'
        e4Hr_path = os.path.join(user_path, days, 'e4Hr', str(int(ts[idx])))+'.csv'
        e4Temp_path = os.path.join(user_path, days, 'e4Temp', str(int(ts[idx])))+'.csv'
        mAcc_path = os.path.join(user_path, days, 'mAcc', str(int(ts[idx])))+'.csv'
        mGps_path = os.path.join(user_path, days, 'mGps', str(int(ts[idx])))+'.csv'
        mGyr_path = os.path.join(user_path, days, 'mGyr', str(int(ts[idx])))+'.csv'
        mMag_path = os.path.join(user_path, days, 'mMag', str(int(ts[idx])))+'.csv'

        
        e4Acc_x, e4Acc_y, e4Acc_z = pre_e4Acc(e4Acc_path,sampling_rate=1920)
        e4Bvp = pre_e4Bvp(e4Bvp_path,sampling_rate=240)
        e4Eda = pre_e4Eda(e4Eda_path,sampling_rate=240)
        e4Hr = pre_e4Hr(e4Hr_path,sampling_rate=240)
        e4Temp = pre_e4Temp(e4Temp_path,sampling_rate=240)
        mAcc_x, mAcc_y, mAcc_z = pre_mAcc(mAcc_path,sampling_rate=1920)
        mGps_lat, mGps_lon, mGps_acc = pre_mGps(mGps_path,sampling_rate=240)
        mGyr_x, mGyr_y, mGyr_z, mGyr_roll, mGyr_pitch, mGyr_yaw = pre_mGyr(mGyr_path,sampling_rate=1920)
        mMag_x, mMag_y, mMag_z = pre_mMag(mMag_path,sampling_rate=1920)

        e4Acc_x_.append(e4Acc_x)
        e4Acc_y_.append(e4Acc_y)
        e4Acc_z_.append(e4Acc_z)
        e4Bvp_.append(e4Bvp)
        e4Eda_.append(e4Eda)
        e4Hr_.append(e4Hr)
        e4Temp_.append(e4Temp)     
        mAcc_x_.append(mAcc_x)
        mAcc_y_.append(mAcc_y)
        mAcc_z_.append(mAcc_z)

        mGps_lat_.append(mGps_lat)
        mGps_lon_.append(mGps_lon)
        mGps_acc_.append(mGps_acc)

        mGyr_x_.append(mGyr_x)
        mGyr_y_.append(mGyr_y)
        mGyr_z_.append(mGyr_z)
        mGyr_roll_.append(mGyr_roll)
        mGyr_pitch_.append(mGyr_pitch)
        mGyr_yaw_.append(mGyr_yaw)

        mMag_x_.append(mMag_x)
        mMag_y_.append(mMag_y)
        mMag_z_.append(mMag_z)

        timestamp, labelaction, labelactionOption = ts[idx], label_action[idx], label_actionOption[idx]
        ts_.append(timestamp)
        label_action_.append(labelaction)
        label_actionOption_.append(labelactionOption)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(user_path.split('user'))
    file_name = os.path.join(save_dir,'user'+user_path.split('user')[-1] + '_' + days + '.npz')
    print('save as', file_name)
    np.savez(file_name, ts = np.stack(ts_),e4Acc_x= np.stack(e4Acc_x_) , e4Acc_y= np.stack(e4Acc_y_), e4Acc_z= np.stack(e4Acc_z_), e4Bvp=np.stack(e4Bvp_), e4Eda=np.stack(e4Eda_), e4Hr=np.stack(e4Hr_), e4Temp=np.stack(e4Temp_), 
                mAcc_x=np.stack(mAcc_x_), mAcc_y=np.stack(mAcc_y_), mAcc_z=np.stack(mAcc_z_), mGps_lat=np.stack(mGps_lat_), mGps_lon=np.stack(mGps_lon_), mGps_acc=np.stack(mGps_acc_), mGyr_x=np.stack(mGyr_x_), mGyr_y=np.stack(mGyr_y_), mGyr_z=np.stack(mGyr_z_),
                mGyr_roll=np.stack(mGyr_roll_), mGyr_pitch=np.stack(mGyr_pitch_), mGyr_yaw=np.stack(mGyr_yaw_), mMag_x=np.stack(mMag_x_), mMag_y=np.stack(mMag_y_), mMag_z=np.stack(mMag_z_), label_action=np.stack(label_action_), label_actionOption=np.stack(label_actionOption_))

def save_timewise_file(user_path, days, save_dir):

    label = pd.read_csv(os.path.join(user_path, days, days +'_label.csv'))
    label_action = label['action']
    label_action = pre_label(label_action)
    label_actionOption = label['actionOption']
    ts = label['ts']

    # e4Acc_x_,e4Acc_y_,e4Acc_z_,e4Bvp_,e4Eda_,e4Hr_,e4Temp_,mAcc_x_, mAcc_y_, mAcc_z_,mGps_lat_, mGps_lon_, mGps_acc_,mGyr_x_, mGyr_y_, mGyr_z_,mGyr_roll_, mGyr_pitch_, mGyr_yaw_, mMag_x_, mMag_y_, mMag_z_= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    # ts_, label_action_, label_actionOption_ = [], [],[]

    for idx in range(len(ts)):
        e4Acc_path = os.path.join(user_path, days, 'e4Acc', str(int(ts[idx])))+'.csv'
        e4Bvp_path = os.path.join(user_path, days, 'e4Bvp', str(int(ts[idx])))+'.csv'
        e4Eda_path = os.path.join(user_path, days, 'e4Eda', str(int(ts[idx])))+'.csv'
        e4Hr_path = os.path.join(user_path, days, 'e4Hr', str(int(ts[idx])))+'.csv'
        e4Temp_path = os.path.join(user_path, days, 'e4Temp', str(int(ts[idx])))+'.csv'
        mAcc_path = os.path.join(user_path, days, 'mAcc', str(int(ts[idx])))+'.csv'
        mGps_path = os.path.join(user_path, days, 'mGps', str(int(ts[idx])))+'.csv'
        mGyr_path = os.path.join(user_path, days, 'mGyr', str(int(ts[idx])))+'.csv'
        mMag_path = os.path.join(user_path, days, 'mMag', str(int(ts[idx])))+'.csv'

        
        e4Acc_x, e4Acc_y, e4Acc_z = pre_e4Acc(e4Acc_path,sampling_rate=1920)
        
        e4Bvp = pre_e4Bvp(e4Bvp_path,sampling_rate=240)
        e4Eda = pre_e4Eda(e4Eda_path,sampling_rate=240)
        e4Hr = pre_e4Hr(e4Hr_path,sampling_rate=240)
        e4Temp = pre_e4Temp(e4Temp_path,sampling_rate=240)
        mAcc_x, mAcc_y, mAcc_z = pre_mAcc(mAcc_path,sampling_rate=1920)
        mGps_lat, mGps_lon, mGps_acc = pre_mGps(mGps_path,sampling_rate=240)
        mGyr_x, mGyr_y, mGyr_z, mGyr_roll, mGyr_pitch, mGyr_yaw = pre_mGyr(mGyr_path,sampling_rate=1920)
        mMag_x, mMag_y, mMag_z = pre_mMag(mMag_path,sampling_rate=1920)


        timestamp, labelaction, labelactionOption = ts[idx], label_action[idx], label_actionOption[idx]


        # print(user_path.split('user'))
        file_name = os.path.join(save_dir,'user'+user_path.split('user')[-1] + '_' + days + '_'+ str(int(timestamp)) + '.npz')
        print('save as', file_name)
        np.savez(file_name, ts = timestamp,e4Acc_x= e4Acc_x , e4Acc_y= e4Acc_y, e4Acc_z= e4Acc_z, e4Bvp=e4Bvp, e4Eda=e4Eda, e4Hr=e4Hr, e4Temp=e4Temp, 
                    mAcc_x=mAcc_x, mAcc_y=mAcc_y, mAcc_z=mAcc_z, mGps_lat=mGps_lat, mGps_lon=mGps_lon, mGps_acc=mGps_acc, mGyr_x=mGyr_x, mGyr_y=mGyr_y, mGyr_z=mGyr_z,
                    mGyr_roll=mGyr_roll, mGyr_pitch=mGyr_pitch, mGyr_yaw=mGyr_yaw, mMag_x=mMag_x, mMag_y=mMag_y, mMag_z=mMag_z, label_action=labelaction, label_actionOption=labelactionOption)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='daywise') # timewise (for timewise preprocessing)
    parser.add_argument('--file_dir', type=str, default='D:/ETRI/data') 
    parser.add_argument('--save_dir', type=str, default='D:/ETRI/preprocessed_daywise') # D:/ETRI/preprocessed_timewise

    return parser.parse_args()



class Preprocess(object):
    def __init__(self, args):
        self.type = args.type
        self.file_dir = args.file_dir
        self.save_dir = args.save_dir
        self.user_list = natsort.natsorted(os.listdir(self.file_dir))


    def process(self):
        if self.type == 'daywise':
            for user in self.user_list:
                user_path = os.path.join(self.file_dir, user)
                print('----------', user_path, '----------')
                user_day = os.listdir(user_path)
                for days in user_day:
                    print(os.path.join(user_path, days))

                    save_daywise_file(user_path, days, self.save_dir)

        if self.type == 'timewise':
            for user in self.user_list:
                user_path = os.path.join(self.file_dir, user)
                print('----------', user_path, '----------')
                user_day = os.listdir(user_path)
                for days in user_day:
                    print(os.path.join(user_path, days))
                    save_timewise_file(user_path, days, self.save_dir)


def main():
    args = parse_args()
    preprocess = Preprocess(args)
    preprocess.process()

if __name__ == '__main__':
    main()