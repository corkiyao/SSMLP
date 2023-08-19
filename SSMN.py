# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 22:47:34 2022

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
from MPutils import MPSNR, SSIM, SAM, testDataset, trainDataset, read_training_data, getModelSize
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x

def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)
        self.apply(self.cls_init_weights)
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AxialShift5(nn.Module):
    def __init__(self, dim, shift_size, droprate=0.1, as_bias=True):
        super().__init__()
        self.dim = dim
        self.shift_size5 = shift_size
        self.pad = self.shift_size5 // 2
        self.conv1 = nn.Linear(dim, dim, bias=as_bias)
        self.conv2_1 = nn.Linear(dim, dim, bias=as_bias)
        self.conv2_2 = nn.Linear(dim, dim, bias=as_bias)
        self.conv3 = nn.Linear(dim, dim, bias=as_bias)
        self.drop = nn.Dropout(p=droprate)
        self.reweight = Mlp(dim, dim//2, dim * 3, droprate)
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        B, H, W, C = x.shape # (N B) H W C
        c = self.conv1(x)
        
        x = x.permute(0, 3, 1, 2)
        # h
        xH = F.pad(x, (0, 0, self.pad, self.pad) , "constant", 0) 
        newx = xH[:, 0:15, :, :]
        x1 = newx[:, 0::5, :, :]
        x2 = newx[:, 1::5, :, :]
        x3 = newx[:, 2::5, :, :]
        x4 = newx[:, 3::5, :, :]
        x5 = newx[:, 4::5, :, :]
        x6 = xH[:, 15:16, :, :]
        
        h1 = torch.narrow(x1, 2, self.pad, H)
        h2 = torch.roll(x2, shifts= 1, dims=2)
        h2 = torch.narrow(h2, 2, self.pad, H)
        h3 = torch.roll(x3, shifts= 2, dims=2)
        h3 = torch.narrow(h3, 2, self.pad, H)
        h4 = torch.roll(x4, shifts= -2, dims=2)
        h4 = torch.narrow(h4, 2, self.pad, H)
        h5 = torch.roll(x5, shifts= -1, dims=2)
        h5 = torch.narrow(h5, 2, self.pad, H)
        h6 = torch.narrow(x6, 2, self.pad, H) #
        
        new  = torch.zeros(B, C - 1, H, W, device="cuda")
        newh  = torch.zeros(B, C, H, W, device="cuda")
        new[:, 0::5, :, :] = h1
        new[:, 1::5, :, :] = h2
        new[:, 2::5, :, :] = h3
        new[:, 3::5, :, :] = h4
        new[:, 4::5, :, :] = h5
        newh[:, 0:15, :, :] = new
        newh[:, 15:16, :, :] = h6
        h = self.conv2_1(newh.permute(0,2,3,1))
        
        # W
        xW = F.pad(x, (self.pad, self.pad, 0, 0) , "constant", 0) #左右填充
        newx = xW[:, 0:15, :, :]
        x1 = newx[:, 0::5, :, :]
        x2 = newx[:, 1::5, :, :]
        x3 = newx[:, 2::5, :, :]
        x4 = newx[:, 3::5, :, :]
        x5 = newx[:, 4::5, :, :]
        x6 = xW[:, 15:16, :, :]
        w1 = torch.narrow(x1, 3, self.pad, W)
        w2 = torch.roll(x2, shifts= 1, dims=3)
        w2 = torch.narrow(w2, 3, self.pad, W)
        w3 = torch.roll(x3, shifts= 2, dims=3)
        w3 = torch.narrow(w3, 3, self.pad, W)
        w4 = torch.roll(x4, shifts= -2, dims=3)
        w4 = torch.narrow(w4, 3, self.pad, W)
        w5 = torch.roll(x5, shifts= -1, dims=3)
        w5 = torch.narrow(w5, 3, self.pad, W)
        w6 = torch.narrow(x6, 3, self.pad, W)
        
        new  = torch.zeros(B, C - 1, H, W, device="cuda")
        newW  = torch.zeros(B, C, H, W, device="cuda")
        new[:, 0::5, :, :] = w1
        new[:, 1::5, :, :] = w2
        new[:, 2::5, :, :] = w3
        new[:, 3::5, :, :] = w4
        new[:, 4::5, :, :] = w5
        newW[:, 0:15, :, :] = new
        newW[:, 15:16, :, :] = w6
        w = self.conv2_2(newW.permute(0,2,3,1))
        
        
        a = (h + w + c).permute(0,3,1,2).flatten(2).mean(2) ## (N B) C
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2) #
        h = h.permute(0,3,1,2) * a[0]
        w = w.permute(0,3,1,2) * a[1]
        c = c.permute(0,3,1,2) * a[2]
        x = (h + w + c).permute(0, 2, 3, 1)
        
        x = self.conv3(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class CycleFC(nn.Module):
    """
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

class CycleMLP(nn.Module):
    def __init__(self, dim, shift_size, qkv_bias=True):
        super().__init__()
        self.mlp_c = nn.Linear(dim , dim , bias=qkv_bias)
        self.sfc_h = CycleFC(dim , dim, (1, shift_size), 1, 0)
        self.sfc_w = CycleFC(dim , dim, (shift_size, 1), 1, 0)
        self.reweight = Mlp(dim , dim , dim  * 3, 0.5)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=0.5)
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        h = self.drop(h)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.drop(w)
        c = self.mlp_c(x)
        c = self.drop(c)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpeBranch(nn.Module):
    def __init__(self, embedim, band, groupSpe, shift, droprate=0.1):
        super(SpeBranch, self).__init__()
        self.shift = shift
        self.embedim = embedim
        self.band = band
        self.groupSpe = groupSpe  
        self.bandglobal = self.band // self.groupSpe  
        if self.band != self.groupSpe * self.bandglobal: print("error or to pad")
        self.shift = shift
        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.head = nn.Linear(embedim, embedim, bias=True)
        self.mlp1 = nn.Linear(embedim//2, embedim//2,bias=True)
        self.mlp21 = nn.Linear(embedim//2 * self.bandglobal, embedim//2 * self.bandglobal, bias=True)
        self.mlp31 = nn.Linear(embedim//2 * self.bandglobal, embedim//2 * self.bandglobal, bias=True)
        self.mlp41 = nn.Linear(embedim//2 * self.groupSpe, embedim//2 * self.groupSpe, bias=True)
        self.concate = nn.Linear(embedim, embedim // 2, bias=True)
        self.fusion = nn.Linear(embedim, embedim, bias=True)
        self.drop = nn.Dropout(p=droprate)
        self.apply(self.cls_init_weights)
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):

        N, B, H, W, C = x.shape
        x = self.head(x)
        x = self.gelu(x)
        x1,x2 = x.chunk(2,dim=4)
        
        # N B H W C1
        x1 = self.mlp1(x1)
        
        x2 = x2.permute(0,4,1,2,3)
        N1, C1, B1, H1, W1 = x2.shape
        
        ##
        x21 = x2.view(N1, C1, self.groupSpe, self.band // self.groupSpe, H1, W1).contiguous()
        x21 = rearrange(x21, " N C g b H W -> N g H W (C b)")
        x21 = self.mlp21(x21)
        x21 = rearrange(x21, " N g H W (C b) -> N C (g b) H W", b=self.bandglobal)
        
        ##shift
        x22 = torch.roll(x2, shifts = -self.shift, dims=2)
        x22 = x22.view(N1, C1, self.groupSpe, self.band // self.groupSpe, H1, W1).contiguous()
        x22 = rearrange(x22, " N C g b H W -> N g H W (C b)")
        x22 = self.mlp31(x22)
        x22 = rearrange(x22, " N g H W (C b) -> N C (g b) H W", b=self.bandglobal)
        x22 = torch.roll(x22, shifts = self.shift, dims=2)
        
        x2 = (torch.cat([x21, x22], dim=1)).permute(0,2,3,4,1)
        x2 = self.concate(x2).permute(0,4,1,2,3)

        ## sparse global
        x4 = x2.view(N1, C1, self.groupSpe, self.band // self.groupSpe, H1, W1).contiguous()
        x4 = rearrange(x4, " N C g b H W -> N b H W (C g) ")
        x4 = self.mlp41(x4)
        x4 = rearrange(x4, " N b H W (C g) -> N (g b) H W C ",g=self.groupSpe)
        
        x = self.fusion(torch.cat([x1,x4],dim=4))
        x = self.drop(x)
        return x

class Spabranch(nn.Module):
    def __init__(self, embedim, band, kernel5, droprate=0.1):
        super(Spabranch, self).__init__()
        self.band = band
        self.head = nn.Linear(embedim, embedim, bias=True)
        
        # self.axial_shift11 = CycleMLP(dim=embedim//4, shift_size=5)
        # self.axial_shift31 = CycleMLP(dim=embedim//4, shift_size=5)
        # self.axial_shift51 = CycleMLP(dim=embedim//4, shift_size=5)
        # self.axial_shift71 = CycleMLP(dim=embedim//4, shift_size=5)
        
        self.axial_shift11 = AxialShift5(dim=embedim//4, shift_size = kernel5)
        self.axial_shift31 = AxialShift5(dim=embedim//4, shift_size = kernel5)
        self.axial_shift51 = AxialShift5(dim=embedim//4, shift_size = kernel5)
        self.axial_shift71 = AxialShift5(dim=embedim//4, shift_size = kernel5)
        
        self.fusion = nn.Linear(embedim, embedim, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(p=droprate)
        
        self.apply(self.cls_init_weights)
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, B, H, W, C = x.shape
        
        x = x.permute(0, 4, 1, 2, 3)
        x = _to_4d_tensor(x, depth_stride=1)
        x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        x = self.gelu(x)
        N1, H1, W1, C1 = x.shape
        
        x = x.view(N1, H1, W1, 4, C1//4).permute(0, 1, 2, 4, 3).contiguous().view(N1, H1, W1, C1)
        x1, x2, x3, x4 = x.chunk(4, dim=3)
        out1 = self.axial_shift11(x1)
        out3 = self.axial_shift31(x2)
        out5 = self.axial_shift51(x3)
        out7 = self.axial_shift71(x4)
        
        x = torch.cat([out1, out3, out5, out7], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fusion(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        x = _to_5d_tensor(x, depth=self.band)
        x = x.permute(0, 2, 3,  4, 1)
        return x

class Cross(nn.Module):
    def __init__(self, wn, embedim, droprate=0.5):
        super(Cross, self).__init__()
        self.mlp1 = nn.Linear(embedim, embedim, bias=True)
        self.mlp2 = nn.Linear(embedim, embedim, bias=True)
        self.mlp3 = nn.Sequential(nn.Linear(embedim, embedim//2, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(embedim//2, embedim, bias=True))
        self.drop = nn.Dropout(p=droprate)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        
        self.apply(self.cls_init_weights)
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        N, B, H, W, C = x.shape
        x1 = self.drop(self.relu(self.mlp1(x)))
        x = self.drop(self.mlp2(x1))
        return x


##Double branch
class DoubleBranch(nn.Module):
    def __init__(self, wn, embedim, groupSpe, shift, band, mlp_ratio=4., drop=0., drop_path=0.,
                 kernel5=5, skip_lam=1.0, droprate = 0.1, if_shared=None):
        super(DoubleBranch, self).__init__()
        self.if_shared = if_shared

        self.Spe = SpeBranch(embedim=embedim, band=band, groupSpe=groupSpe,
                             shift=shift, drop_path=drop_path, droprate=droprate)
        self.Spa = Spabranch(wn, embedim=embedim, band=band,
                             kernel5=kernel5, droprate=droprate)
        
        self.fusion = nn.Linear(embedim, embedim, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop = nn.Dropout(p=droprate)
        self.norm1 = nn.LayerNorm(embedim)
        self.norm2 = nn.LayerNorm(embedim)
        self.norm3 = nn.LayerNorm(embedim)
        self.gelu = nn.GELU()
        self.ffn = Cross(wn, embedim, band, droprate=droprate)
        
        self.apply(self.cls_init_weights)
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, B, H, W, C = x.shape

        ##attention
        xT = self.Spe(self.norm1(x)) + x
        xS = self.Spa(self.norm2(xT)) + x
        
        ##ffn
        x = self.ffn(self.norm3(xS)) + xS
        return x

##whole model
class MLPours(nn.Module):
    def __init__(self, inchannel=1, embedim=[32,64], groupSpe=8, shift=[0,2,3], 
                 kernel3=3, kernel5=5, kernel7=7, layers=[1,1,1,1,1,1], start=0, end=64, middle=16,
                 band=48, scale=2, mlp_ratio=[4],drop=0., downscale=2, upscale=2,
                 drop_path_rate=0.1,attn_drop=0., drop_path=0., skip_lam=1.0, droprate=0.3,if_shared=None):
        super(MLPours, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.scale = scale
        self.band= band
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.head = nn.Linear(inchannel, embedim[1],bias=True)
        
        self.block1 = nn.ModuleList([])
        for i in range(layers[0]):
            self.block1.append(
                DoubleBranch(wn,embedim=embedim[1], groupSpe=groupSpe,shift=shift[0],band=band,
                             drop=drop,mlp_ratio=mlp_ratio[0],attn_drop=attn_drop, drop_path=dpr[i], 
                             skip_lam=skip_lam, kernel3=kernel3, kernel5=kernel5, kernel7=kernel7, start=start, 
                             end=end, middle=middle, droprate=droprate, if_shared=if_shared))
            
        self.block2 = nn.ModuleList([])
        for i in range(layers[1]):
            self.block2.append(
                DoubleBranch(wn,embedim=embedim[1], groupSpe=groupSpe, shift=shift[0],band=band,
                             drop=drop,mlp_ratio=mlp_ratio[0],attn_drop=attn_drop, drop_path=dpr[i], 
                             skip_lam=skip_lam, kernel3=kernel3, kernel5=kernel5, kernel7=kernel7, start=start, 
                             end=end, middle=middle, droprate=droprate, if_shared=if_shared))
            
        self.block3 = nn.ModuleList([])
        for i in range(layers[2]):
            self.block3.append(
                DoubleBranch(wn,embedim=embedim[1], groupSpe=groupSpe, shift=shift[0],band=band,
                             drop=drop,mlp_ratio=mlp_ratio[0],attn_drop=attn_drop, drop_path=dpr[i], 
                             skip_lam=skip_lam, kernel3=kernel3, kernel5=kernel5, kernel7=kernel7, start=start, 
                             end=end, middle=middle, droprate=droprate, if_shared=if_shared))
        
        self.block4 = nn.ModuleList([])
        for i in range(layers[3]):
            self.block4.append(
                DoubleBranch(wn,embedim=embedim[1], groupSpe=groupSpe, shift=shift[0],band=band,
                             drop=drop,mlp_ratio=mlp_ratio[0],attn_drop=attn_drop, drop_path=dpr[i], 
                             skip_lam=skip_lam, kernel3=kernel3, kernel5=kernel5, kernel7=kernel7, start=start, 
                             end=end, middle=middle, droprate=droprate, if_shared=if_shared))
        

        self.up1 = nn.Linear(embedim[1], embedim[1], bias=True)
        self.up2 = nn.Linear(embedim[1], embedim[1] * self.scale * self.scale, bias=True)
        self.up3 = nn.Linear(self.band, self.band, bias=True)
        self.end = nn.Linear(embedim[1], inchannel, bias=True)
        self.drop = nn.Dropout(p=droprate)
        self.gelu = nn.GELU()
        self.headup = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
        self.relu = nn.ReLU(inplace=True)

        self.apply(self.cls_init_weights)
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.headup(x)
        x = x.unsqueeze(1)
        
        x = self.head(x.permute(0,2,3,4,1))
        shortcut1 = x
        
        for blk1 in self.block1:
            x = blk1(x)
        x1 = x
        
        for blk2 in self.block2:
            x = blk2(x)
        x2 = x

        for blk3 in self.block3:
            x = blk3(x)
        x3 = x
        
        for blk4 in self.block4:
            x = blk4(x)
        x4 = x
        
        x5 = self.up1(x1 + x2 + x3 + x4 + shortcut1)
        x5 = self.drop(x5)
        N, B, H, W, C = x5.shape
        oH, oW = self.scale * H, self.scale * W 
        
        x = x5.permute(0,4,1,2,3)
        x = _to_4d_tensor(x, depth_stride=1)
        
        N1, C1, H1, W1 = x.shape
        x = x.permute(0,2,3,1)
        x = self.up2(x).contiguous().view(N1, H, W, C1, self.scale, self.scale)
        x = x.permute(0,1,4,2,5,6).contiguous()
        x = x.reshape(N1, oH, oW, C1)
        
        x = x.permute(0,3,1,2)
        x = _to_5d_tensor(x, depth=self.band)
        x = self.end(x.permute(0,2,3,4,1)) 
        x = x.squeeze(4) 
        x = self.up3(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = x + shortcut
        return x


def train1(model, EPOCH):
    model = model
    accum_steps = 1
    train_data, train_label = read_training_data(
        "./HoustonU4_12_48.h5")
    print(train_data.shape, train_label.shape)
    deal_dataset = trainDataset(train_data, train_label)
    train_loader = DataLoader.DataLoader(deal_dataset, batch_size = 24, shuffle=True)  
    
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5, last_epoch=-1)
    
    model.train()
    loss_list = []
    for eopch in range(EPOCH):
        for step, data in enumerate(train_loader):    
            loss = nn.L1Loss() 
            train_data, train_label = data
            train_data, train_label = Variable(train_data.cuda()), Variable(train_label.cuda())
            output = model(train_data)
            loss1 = loss(output, train_label)
            if step % 200 == 0:
                print('4beiEpoch:', eopch, 'Step: ', step, 
                      'loss: {:.6f}\t'.format(float(loss1)))
                loss_list.append(float(loss1))
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
        if eopch < 200:
            torch.save(model.state_dict(),
                       './MLP-4bei_%d.pkl' % (eopch))
        scheduler.step()
    return loss_list

def test1(model, start, end):
    h5File = h5py.File('./houstonU4_36-144.mat', 'r')
    data = h5File['dataa']
    label = h5File['label']
    print(label.shape) 
    
    data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2], data.shape[3]])
    label = np.reshape(label, [label.shape[0], label.shape[1], label.shape[2], label.shape[3]])
    
    deal_dataset = testDataset(data, label)
    test_loader = DataLoader.DataLoader(deal_dataset, batch_size=1, shuffle=False)
    for z in range(start, end):
            print(z)
            GDRRN1 = model
            GDRRN1.eval()
            GDRRN1.load_state_dict(
                torch.load('./MLP-4bei_%d.pkl' %(z) ))
            
            psnr, ssim, sam=0.0, 0.0, 0.0
            output12=[] 
            torch.cuda.synchronize()
            start = time.time()
            for i, (image, label) in enumerate(test_loader):
                image, label = image.cuda(), label.cuda()
                image, label = Variable(image), Variable(label)
                with torch.no_grad():
                    SR = GDRRN1(image)  
                
                SR = SR.cpu().data[0].numpy().astype(np.float32) 
                SR = np.array(np.transpose(SR, [2, 1, 0]))
                
                SR1 = np.reshape(SR, [SR.shape[0], SR.shape[1], SR.shape[2], 1])
                output12.append(SR1)
                result12 = np.concatenate(output12, axis=3)
                
                label=label.cpu().data[0].numpy().astype(np.float32)
                label = np.array(np.transpose(label, [2, 1, 0]))
                
                psnr = psnr + MPSNR(label, SR)
                ssim = ssim + SSIM(label, SR)
                sam = sam + SAM(label, SR)
            torch.cuda.synchronize()
            time_sum = time.time() - start
            print("MLPeach time = ", time_sum / len(test_loader))
            print('psnr', psnr / len(test_loader))
            print('sssim', ssim / len(test_loader)) 
            print('ssam', sam / len(test_loader))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed = 3047
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    
    import gc
    gc.collect()
    model = MLPours(scale=4, band=48, kernel3=3, kernel5=5, kernel7=7, start=0, end=64, middle=16, droprate=0.2)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    #train1(model, 50)    
    #test1(model, 15, 50)
    a = torch.randn(2,48,36,36).cuda()
    b = model(a)
    print("final shape",b.shape)
    





