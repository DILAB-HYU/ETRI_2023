import tsaug
import random
import numpy as np

def add_noise_1(x):
    return tsaug.AddNoise(scale=0.01).augment(x)

def add_noise_2(x):
    return tsaug.AddNoise(scale=0.05).augment(x)

def scale_1(x):
    return x * np.random.normal(0.1, 0.01, x.shape[0])

def scale_2(x):
    return x * np.random.normal(0.1, 0.05, x.shape[0])

def quantize(x):
    return tsaug.Quantize(n_levels=20).augment(x)

def convolve(x):
    return tsaug.Convolve(window="flattop", size=20).augment(x)

def reverse(x):
    return tsaug.Reverse().augment(x)

def timewarp(x):
    return tsaug.TimeWarp(n_speed_change=20, max_speed_ratio=10).augment(x)

def drift(x):
    return tsaug.Drift(max_drift=0.5, n_drift_points=3).augment(x)

# Augmentation. Randomly selected from the list.
def aug(x, augmentations = [add_noise_1, add_noise_2, scale_1, scale_2, quantize, convolve, reverse, timewarp, drift]):
    random_list = [1] + [random.randint(0,1) for _ in range(8)]
    random.shuffle(random_list)
    random_list

    for i in range(9):
        if random_list[i] == 1:
            # print(augmentations[i])
            output = augmentations[i](x)
    return output


file = np.load(file_name)
for i in range(n):
    for file in label_carehousemem:
        data = np.load(os.path.join(path_dir, file))

        np.savez(os.path.join(output_dir, file+'_aug_'+str(i+1)),
        ts = data['ts'],e4Acc_x= aug(data['e4Acc_x']) , e4Acc_y= aug(data['e4Acc_y']), e4Acc_z= aug(data['e4Acc_z']), e4Bvp=aug(data['e4Bvp']), e4Eda=aug(data['e4Eda']), e4Hr=aug(data['e4Hr']), e4Temp=aug(data['e4Temp']), 
                    mAcc_x=aug(data['mAcc_x']), mAcc_y=aug(data['mAcc_y']), mAcc_z=aug(data['mAcc_z']), mGps_lat=data['mGps_lat'], mGps_lon=data['mGps_lon'], mGps_acc=data['mGps_acc'], mGyr_x=aug(data['mGyr_x']), mGyr_y=aug(data['mGyr_y']), mGyr_z=aug(data['mGyr_z']),
                    mGyr_roll=aug(data['mGyr_roll']), mGyr_pitch=aug(data['mGyr_pitch']), mGyr_yaw=aug(data['mGyr_yaw']), mMag_x=aug(data['mMag_x']), mMag_y=aug(data['mMag_y']), mMag_z=aug(data['mMag_z']), label_action=data['label_action'], label_actionOption=data['label_actionOption'])