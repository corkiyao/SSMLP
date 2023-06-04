# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:13:43 2022

@author: admin
"""
import time
import torch
import gc
import h5py
import random
import os
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import init
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import scipy.io as sio
from sklearn.metrics import  mean_squared_error

def MPSNR(x_true, x_pred):
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true=x_true[:,:,:]
    for k in range(n_bands):
        x_true_k = x_true[  :, :, k].reshape([-1])
        x_pred_k = x_pred[  :, :, k,].reshape([-1])
        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )
        MAX_k = np.max(x_true_k)
        if MAX_k != 0 :
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
        else:
            mask[k] = 0
    psnr = PSNR.sum() / mask.sum()
    mse = MSE.mean()
    return psnr

def SSIM(x_true,x_pre):
    num=x_true.shape[2]
    ssimm=np.zeros(num)
    c1=0.0001
    c2=0.0009
    n=0
    for x in range(x_true.shape[2]):
            z = np.reshape(x_pre[:, :,x], [-1])
            sa=np.reshape(x_true[:,:,x],[-1])
            y=[z,sa]
            cov=np.cov(y)
            oz=cov[0,0]
            osa=cov[1,1]
            ozsa=cov[0,1]
            ez=np.mean(z)
            esa=np.mean(sa)
            ssimm[n]=((2*ez*esa+c1)*(2*ozsa+c2))/((ez*ez+esa*esa+c1)*(oz+osa+c2))
            n=n+1
    SSIM=np.mean(ssimm)
    return SSIM  

def SAM(x_true,x_pre):
    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[ x, y,:], [-1])
            sa = np.reshape(x_true[x, y,:], [-1])
            tem1=np.dot(z,sa)
            tem2=(np.linalg.norm(z))*(np.linalg.norm(sa))
            A=(tem1+0.0001)/(tem2+0.0001)
            if A>1:
                A=1
            samm[n]=np.arccos(A)
            n=n+1
    SAM=(np.mean(samm))*180/np.pi
    return SAM

class testDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label
    
class trainDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label

def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = data
        train_label = label
        return train_data, train_label

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('model size：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

