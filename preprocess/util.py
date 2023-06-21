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



###### segmentation visualization ###### 
def user_segmentation_vis(predlist, lbllist, sensor1, sensor2,
                        vis_title = 'user 07', 
                        line_width=0.8, 
                        mode = 'label_all', vis_label=False):

    '''
    Segmentation 결과 visualization 하는 함수 

    Args: 
        - predlist [array] : model prediction output e.g., pred.cpu().detach().numpy() 
        - lbllist [array] : ground truth 
        - vis_title [str]: plot title 
        - line_width [float] : plot line width. 1에 가까울수록 두꺼움. 
        - mode [label_all , label_one]: segentation결과 label을 모든 segmentation에 대해 display할지, 아니면 겹치지 않게 1개씩만 display할지 여부 
        - vis_label [bool] : label visualization 할 지 여부 
    '''

    category_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    predlist_ = predlist.cpu().detach().numpy() 
    lbllist_ = lbllist.cpu().detach().numpy() 
    ##### 각 클래스별 색상 생성 
    category_colors = plt.get_cmap('Pastel2')(np.linspace(0.2, 0.8, len(category_names)))
    print(np.shape(category_colors))
    color_list = []
    for i in range(len(predlist_)):
        color = category_colors[predlist_[i]]
        color_list.append(color)

    ###### plot 생성  
    fig, ax = plt.subplots(figsize=(20, 5))

    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, predlist_.shape[0])

    for i, (colname, color) in enumerate(zip(predlist_, color_list)):
            widths = line_width 
            starts = i 
            ax.barh(vis_title, widths, left=starts, height=0.001,
                            label = colname, color = color)
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'black'
            ax.plot(x[0,i,:])

    ##### label print 하고 싶을 경우에만 
    if vis_label:
        for i, (rect)in enumerate(ax.patches):

                y_value = 0.00005 
                if predlist_[i] == lbllist_[i]:text_color = 'blue'
                else: text_color = 'red'

                ####### display할 label에 따라 xaxis 지정. 
                if mode == 'label_all': ### 모든 label display 
                    x_value = rect.get_x() + rect.get_width() / 2 - 0.5
                    ax.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=True)

                else: ### 겹치지 않게 한개의 label만 display 
                    if i<127 and predlist_[i] == predlist_[i+1]:
                        if predlist_[i] == predlist_[i-1]:
                            x_value = rect.get_x() + rect.get_width() / 2 + 0.5
                            ax.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=False)
                        else:   
                            x_value = rect.get_x() + rect.get_width() / 2 - 1
                            ax.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=False)
                    else:
                        if i == 127 : 
                            x_value = rect.get_x() + rect.get_width() / 2 - 0.5
                            ax.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=True)
                        else: 
                            x_value = rect.get_x() + rect.get_width() / 2 - 1
                            ax.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=True)
    plt.show()


def get_pred(device, f_name, n_iter):
    ################# data load 
    sensor_1 = ['e4Bvp', 'e4Eda', 'e4Hr', 'e4Temp']
    sensor_2 = ['e4Acc_x', 'e4Acc_y', 'e4Acc_z', 'mAcc_x', 'mAcc_y', 'mAcc_z', 
            'mGyr_x', 'mGyr_y', 'mGyr_z', 'mGyr_roll', 'mGyr_pitch', 'mGyr_yaw', 
            'mMag_x', 'mMag_y', 'mMag_z']

    file = np.load(os.path.join(test_dir, f_name))

    x1 = []
    x1 = ([file[ch] for ch in sensor_1])
    x1 = np.stack(x1, axis=0)
    x1 = std_by_channel(x1)
    x1 = torch.from_numpy(x1)
    x2 = []
    x2 = ([file[ch] for ch in sensor_2])
    x2 = np.stack(x2, axis=0)

    x2 = std_by_channel(x2)
    x2 = torch.from_numpy(x2)


    action_label = file['label_action']
    x1_ = x1.permute(1,0,2)
    x2_ = x2.permute(1,0,2)
    y_ = action_label.reshape(-1)

    x1_ = x1_.type(torch.FloatTensor).squeeze(0)
    x2_ = x2_.type(torch.FloatTensor).squeeze(0)
    y_ =  torch.LongTensor(y_).squeeze(0)



    ################## Prediction 
    predlist=torch.zeros(0, dtype=torch.long, device=device)
    lbllist=torch.zeros(0, dtype=torch.long, device=device)  

    batch_loader = segmentation_loader.batchloader(x1_, x2_, y_, y_, batch_size = 128)                
    for iter, (b_x, b_x2, b_target, _) in enumerate(batch_loader):
        if iter == n_iter:
            body_out, body_pred = body_spatio(b_x, edge_index = body_edge) 
            action_out, aciton_pred = action_spatio(b_x2, edge_index = action_edge) 

            lin_in = torch.cat([body_out , action_out], 1)
            pred = linear(lin_in)

            _, predictions = torch.max(pred,-1)
            #print(pred)
            #print("Batch accuracy:", accuracy_score(predictions, b_target))
            predlist = torch.cat([predlist, predictions.view(-1)])
            lbllist = torch.cat([lbllist, b_target.view(-1)])

            predlist, lbllist = predlist.to('cpu'), lbllist.to('cpu')
            acc = accuracy_score(lbllist, predlist)

            f1 = f1_score( lbllist, predlist, average='weighted')
            class_f1 = f1_score( lbllist, predlist, average=None)
            report = classification_report(lbllist, predlist, output_dict = False)
            #print(acc)
            #print(f1)
            #print(report)       

    return predlist, lbllist, x1, x2
