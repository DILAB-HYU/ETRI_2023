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




###### segmentation visualization ###### 
def user_segmentation_vis(predlist, lbllist, x=None, x2=None,
                        vis_title ='user 07', 
                        line_width=0.8, 
                        mode = 'label_all', vis_label = True):

    '''
    Segmentation 결과 visualization 하는 함수 (논문 figure)

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
    color_list = []
    for i in range(len(predlist_)):
        color = category_colors[predlist_[i]]
        color_list.append(color)

    lbl_color_list = []
    for i in range(len(lbllist_)):
        color = category_colors[lbllist_[i]]
        lbl_color_list.append(color)


    ###### plot 생성  
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(20, 5))

    # raw signal 
    ax1.xaxis.set_visible(True)
    ax1.set_xlim(0, predlist_.shape[0])

    # prediction result 
    ax2.invert_yaxis()
    ax2.xaxis.set_visible(True)
    ax2.set_xlim(0, predlist_.shape[0])

    # ground truth 
    ax3.invert_yaxis()
    ax3.xaxis.set_visible(True)
    ax3.set_xlim(0, predlist_.shape[0])

    for i, (colname, color, lbl_color) in enumerate(zip(predlist_, color_list, lbl_color_list)):
            widths = line_width 
            starts = i 
            ax1.plot(x[0,i,:], alpha = 0.15)
            ax1.set_yticks([0])
            ax1.set_yticklabels(labels=[vis_title + '\n' +  'Raw Signal'])    
            #ax1.set_ylabel(vis_title + '\n' +  'Raw Signal')
            if i>0 and predlist_[i-1] != predlist_[i]:
                ax1.axvline(x = (i), color='k', linestyle='--', linewidth=1, ymin=int(x.min()), ymax=int(x.max()), clip_on=True)

            ax2.barh(vis_title + '\n' +  'Prediction', widths, left=starts, height=0.001,
                            label = colname, color = color)
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'black'

            ax3.barh(vis_title + '\n' + "Ground Truth", widths, left=starts, height=0.001,
                            label = colname, color = lbl_color)

            
            
            

    ##### label print 하고 싶을 경우에만 
    if vis_label:
        for i, (rect)in enumerate(ax2.patches):

                y_value = 0.00005 
                if predlist_[i] == lbllist_[i]:text_color = 'blue'
                else: text_color = 'red'

                ####### display할 label에 따라 xaxis 지정. 
                if mode == 'label_all': ### 모든 label display 
                    x_value = rect.get_x() + rect.get_width() / 2 - 0.5
                    ax2.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=True)

                else: ### 겹치지 않게 한개의 label만 display 
                    if i<127 and predlist_[i] == predlist_[i+1]:
                        if predlist_[i] == predlist_[i-1]:
                            x_value = rect.get_x() + rect.get_width() / 2 + 0.5
                            ax2.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=False)
                        else:   
                            x_value = rect.get_x() + rect.get_width() / 2 - 1
                            ax2.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=False)
                    else:
                        if i == 127 : 
                            x_value = rect.get_x() + rect.get_width() / 2 - 0.5
                            ax2.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=True)
                        else: 
                            x_value = rect.get_x() + rect.get_width() / 2 - 1
                            ax2.text(x_value, y_value, predlist_[i], fontname='monospace', color=text_color, linespacing=2,visible=True)
    plt.show()

def user_segmentation_vis_v1(predlist, lbllist, sensor1, sensor2,
                        vis_title = 'user 07', 
                        line_width=0.8, 
                        mode = 'label_all', vis_label=False):

    '''
    Segmentation 결과 visualization 하는 함수 (NOT USED)

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
