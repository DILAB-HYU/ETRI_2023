import numpy as np
import pandas as pd
from scipy import signal

def pre_e4Acc(path, sampling_rate=1920):
    try:
        e4acc = pd.read_csv(path)
        e4Acc_x, e4Acc_y, e4Acc_z = signal.resample(e4acc['x'],sampling_rate), signal.resample(e4acc['y'],sampling_rate), signal.resample(e4acc['z'],sampling_rate)
    except FileNotFoundError:
        e4Acc_x, e4Acc_y, e4Acc_z = np.zeros(sampling_rate), np.zeros(sampling_rate), np.zeros(sampling_rate)
    return e4Acc_x, e4Acc_y, e4Acc_z

def pre_e4Bvp(path, sampling_rate=3840):
    try:
        e4Bvp = pd.read_csv(path)
        e4Bvp = signal.resample(e4Bvp['value'],sampling_rate)
    except FileNotFoundError:
        e4Bvp = np.zeros(sampling_rate)
    return e4Bvp

def pre_e4Eda(path, sampling_rate=240):
    try:
        e4Eda = pd.read_csv(path)
        e4Eda = signal.resample(e4Eda['eda'],sampling_rate)
    except FileNotFoundError:
        e4Eda = np.zeros(sampling_rate)
    return e4Eda

def pre_e4Hr(path, sampling_rate=60):
    try:
        e4Hr = pd.read_csv(path)
        e4Hr = signal.resample(e4Hr['hr'],sampling_rate)
    except FileNotFoundError:
        e4Hr = np.zeros(sampling_rate)
    return e4Hr

def pre_e4Temp(path, sampling_rate=240):
    try:
        e4Temp = pd.read_csv(path)
        e4Temp = signal.resample(e4Temp['temp'],sampling_rate)
    except FileNotFoundError:
        e4Temp = np.zeros(sampling_rate)
    return e4Temp


def pre_mAcc(path, sampling_rate=1800):
    try:
        mAcc = pd.read_csv(path)
        mAcc_x, mAcc_y, mAcc_z = signal.resample(mAcc['x'],sampling_rate), signal.resample(mAcc['y'],sampling_rate), signal.resample(mAcc['z'],sampling_rate)
    except FileNotFoundError:
        mAcc_x, mAcc_y, mAcc_z = np.zeros(sampling_rate), np.zeros(sampling_rate), np.zeros(sampling_rate)
    return mAcc_x, mAcc_y, mAcc_z 


def pre_mGps(path, sampling_rate=12):
    try:
        mGps = pd.read_csv(path)
        mGps_lat, mGps_lon, mGps_acc = signal.resample(mGps['lat'],sampling_rate), signal.resample(mGps['lon'],sampling_rate), signal.resample(mGps['accuracy'],sampling_rate)
    except FileNotFoundError:
        mGps_lat, mGps_lon, mGps_acc = np.zeros(sampling_rate), np.zeros(sampling_rate), np.zeros(sampling_rate)
    return mGps_lat, mGps_lon, mGps_acc
    
def pre_mGyr(path, sampling_rate=1800):
    try:
        mGyr = pd.read_csv(path)
        mGyr_lat, mGyr_long, mGyr_acc = signal.resample(mGyr['x'],sampling_rate), signal.resample(mGyr['y'],sampling_rate), signal.resample(mGyr['z'],sampling_rate)
        mGyr_roll, mGyr_pitch, mGyr_yaw = signal.resample(mGyr['roll'],sampling_rate), signal.resample(mGyr['pitch'],sampling_rate), signal.resample(mGyr['yaw'],sampling_rate)
    except FileNotFoundError:
        mGyr_lat, mGyr_long, mGyr_acc = np.zeros(sampling_rate), np.zeros(sampling_rate), np.zeros(sampling_rate)
        mGyr_roll, mGyr_pitch, mGyr_yaw = np.zeros(sampling_rate), np.zeros(sampling_rate), np.zeros(sampling_rate)

    return mGyr_lat, mGyr_long, mGyr_acc, mGyr_roll, mGyr_pitch, mGyr_yaw
    
def pre_mMag(path, sampling_rate=1800):
    try:
        mMag = pd.read_csv(path)
        mMag_x, mMag_y, mMag_z = signal.resample(mMag['x'],sampling_rate), signal.resample(mMag['y'],sampling_rate), signal.resample(mMag['z'],sampling_rate)
    except FileNotFoundError:
        mMag_x, mMag_y, mMag_z = np.zeros(sampling_rate), np.zeros(sampling_rate), np.zeros(sampling_rate)
    return mMag_x, mMag_y, mMag_z
    
def pre_label(target):
    data_dic = {'label' : ['sleep', 'personal_care', 'work', 'study', 'household', 'care_housemem', 'recreation_media', 'entertainment', 'outdoor_act', 'hobby', 'recreation_etc', 'shop', 'community_interaction', 'travel', 'meal', 'socialising']}
    label_list = data_dic['label']
    target_indices = []
    for label in target:
        index = label_list.index(label)
        target_indices.append(index)
    return target_indices


