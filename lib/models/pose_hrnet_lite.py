# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

import torch.nn.functional as F
import torchvision
 

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#############################################################################
def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)
#############################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True) # 2.
        self.relu = HardSwish(inplace=True) # 2.

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True) # 3.
        self.relu = HardSwish(inplace=True) # 3.

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# class GAU(nn.Module):
#     def __init__(
#         self,
#         dim,
#         query_key_dim = 128,
#         expansion_factor = 2.,
#         add_residual = True,
#         dropout = 0.,
#     ):
#         super().__init__()
#         hidden_dim = int(expansion_factor * dim)

#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)

#         self.to_hidden = nn.Sequential(
#             nn.Linear(dim, hidden_dim * 2),
#             nn.SiLU()
#         )

#         self.to_qk = nn.Sequential(
#             nn.Linear(dim, query_key_dim),
#             nn.SiLU()
#         )
#         self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
#         self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
#         nn.init.normal_(self.gamma, std=0.02)


#         self.to_out = nn.Sequential(
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#         self.add_residual = add_residual

#     def forward(self, x):
#         seq_len = x.shape[-2]

#         normed_x = self.norm(x) #(bs,seq_len,dim)
#         v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #(bs,seq_len,seq_len)

#         Z = self.to_qk(normed_x) #(bs,seq_len,query_key_dim)

#         QK = einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
#         q, k = QK.unbind(dim=-2)

#         sim = einsum('b i d, b j d -> b i j', q, k) / seq_len

#         A = F.relu(sim) ** 2
#         A = self.dropout(A)

#         V = einsum('b i j, b j d -> b i d', A, v)
#         V = V * gate

#         out = self.to_out(V)

#         if self.add_residual:
#             out = out + x

#         return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        # self.relu = nn.ReLU(True)
        self.relu = HardSwish(inplace=True) # 4.
        

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    # nn.ReLU(True)
                                    HardSwish(inplace=True) # 5.
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class SpatialAttentionModule(nn.Module):
    def __init__(self,output_chl_num):
        super(SpatialAttentionModule, self).__init__()
        
        self.conv_ = nn.Conv2d(in_channels=output_chl_num, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()
        self.conv_bn_relu_prm_3_1 = conv_bn_relu(output_chl_num, output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=False)
        self.conv_bn_relu_prm_3_2 = conv_bn_relu(output_chl_num, output_chl_num, kernel_size=9,
                stride=1, padding=4, has_bn=True, has_relu=True,
                efficient=False,groups=output_chl_num)
        self.sigmoid3 = nn.Sigmoid()
        self.conv2d = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.cross = Block(1)
        # self.patch_embed = PatchEmbed(
        #         64, patch_size=16, in_chans=1, embed_dim=768, ratio=1)
    
    def forward(self, x, x_):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        #print(avgout.shape) # torch.Size([2, 1, 64, 48])
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out)) # 这里要改成 2 -> 1

        out_prm = self.conv_bn_relu_prm_3_1(x)
        out_prm = self.conv_bn_relu_prm_3_2(out_prm)
        out_prm = self.sigmoid3(out_prm)
        #print(out_prm.shape)
        final_out = (out + out_prm) / 3
        #print(final_out.shape)
        return final_out
    
 
class MS_CAM(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
 
        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # senet中池化
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        self.sigmoid = nn.Sigmoid()
 
    # x2是深层特征
    def forward(self, x, x2):
        xl = self.local_att(x)
        xg = self.global_att(x2)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

# 原来的MS_CAM2
class MS_CAM2(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
 
        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # senet中池化
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
 
        self.sigmoid = nn.Sigmoid()
 
    # x2是深层特征
    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei
    
# from torch.utils.checkpoint import checkpoint
# class conv_bn_relu(nn.Module):

#     def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
#             has_bn=True, has_relu=True, efficient=False,groups=1):
#         super(conv_bn_relu, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
#                 stride=stride, padding=padding,groups=groups)
#         self.has_bn = has_bn
#         self.has_relu = has_relu
#         self.efficient = efficient
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         def _func_factory(conv, bn, relu, has_bn, has_relu):
#             def func(x):
#                 x = conv(x)
#                 if has_bn:
#                     x = bn(x)
#                 if has_relu:
#                     x = relu(x)
#                 return x
#             return func 

#         func = _func_factory(
#                 self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

#         if self.efficient:
#             x = checkpoint(func, x)
#         else:
#             x = func(x)

#         return x
    
# class SpatialAttentionModule(nn.Module):
#     def __init__(self,output_chl_num):
#         super(SpatialAttentionModule, self).__init__()
                
#         self.sigmoid = nn.Sigmoid()
#         self.conv_bn_relu_prm_3_1 = conv_bn_relu(output_chl_num, output_chl_num, kernel_size=1,
#                 stride=1, padding=0, has_bn=True, has_relu=True,
#                 efficient=False)
#         self.conv_bn_relu_prm_3_2 = conv_bn_relu(output_chl_num, output_chl_num, kernel_size=9,
#                 stride=1, padding=4, has_bn=True, has_relu=True,
#                 efficient=False,groups=output_chl_num)
#         self.sigmoid3 = nn.Sigmoid()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         # self.cross = Block(1)
#         # self.patch_embed = PatchEmbed(
#         #         64, patch_size=16, in_chans=1, embed_dim=768, ratio=1)
    
#     def forward(self, x, x_):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         #print(avgout.shape) # torch.Size([2, 1, 64, 48])
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out)) # 这里要改成 2 -> 1

#         out_prm = self.conv_bn_relu_prm_3_1(x)
#         out_prm = self.conv_bn_relu_prm_3_2(out_prm)
#         out_prm = self.sigmoid3(out_prm)
#         #print(out_prm.shape)
#         final_out = (out + out_prm) / 2
#         #print(final_out.shape)
#         return final_out

class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        assert  cfg.MODEL.COORD_REPRESENTATION in ['simdr', 'sa-simdr', 'heatmap'], 'only simdr and sa-simdr and heatmap supported for pose_resnet_upfree'
        

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = HardSwish(inplace=True) # 6.
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, multi_scale_output=False)


        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        self.mscam32 = MS_CAM(32)
        self.mscam64 = MS_CAM(64)
        self.mscam128 = MS_CAM(128)
        # head
        if self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
            self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            HardSwish(inplace=True) # 3.
                            # nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            # nn.ReLU(inplace=True)
                            HardSwish(inplace=True) # 3.
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        print(x_list.shape)
        y_list = self.stage3(x_list)

        # y_list[0] = self.mscam32(y_list[0],y_list[2])
        # y_list[1] = self.mscam32(y_list[1],y_list[2])

        x_ = self.final_layer(y_list[0])

        if self.coord_representation == 'heatmap':
            return x_
        elif self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':

            # grid = make_grid(x_[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=2, normalize=True, pad_value=0)
            # image_grid = Image.fromarray(grid.mul(255).permute(1,2,0).byte().numpy())
            # image_grid.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")

            x = rearrange(x_, 'b c h w -> b c (h w)')
            pred_x = self.mlp_head_x(x)
            pred_y = self.mlp_head_y(x)
            return pred_x, pred_y

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
