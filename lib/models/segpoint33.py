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
import numpy as np

import torch
import torch.nn as nn
import torchvision
from einops import rearrange, repeat
import torch.nn.functional as F
from transformers.activations import get_activation
INF = float("inf")
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None,):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim

#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
#         self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
#         self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

#     def forward(self, x, **kwargs):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         Hp, Wp = x.shape[2], x.shape[3]

#         x = x.flatten(2).transpose(1, 2)
#         return x, (Hp, Wp)

# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
#                  norm_layer=nn.LayerNorm, attn_head_dim=None
#                  ):
#         super().__init__()
        
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
#             )

#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(True)

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
                                    nn.ReLU(True)
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


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # B C 1 1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

from torch.utils.checkpoint import checkpoint
class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False,groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding,groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x

def attention_normalize(a, l, dim=-1, method="softmax", scaling_factor="n"):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == "softmax":
        return torch.softmax(a, dim=dim)
    else:
        if method == "squared_relu":
            if scaling_factor == "n^2":
                return torch.relu(a / l) ** 2
            elif scaling_factor == "ns":
                return torch.relu(a) ** 2 / (128 * l)
            elif scaling_factor == "scale":
                return torch.relu(a) ** 2 / (128 * l * torch.sum(torch.relu(a) ** 2, dim=-1, keepdim=True))
        elif method == "softmax_plus":
            return torch.softmax(a * torch.log(torch.tensor(l,dtype=torch.float)) / np.log(512), dim=dim)
    return a


class ScaleOffset(nn.Module):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
    """
    
    def __init__(
            self,
            hidden_size=768,
            scale=True,
            offset=True,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset
        
        if self.scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if self.offset:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, inputs):
        if self.scale:
            inputs = inputs * self.weight
        if self.offset:
            inputs = inputs + self.bias
        
        return inputs


class Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        variance = torch.mean(torch.mul(x, x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


class GatedAttentionUnit(nn.Module):
    """门控注意力单元
    链接:https://arxiv.org/abs/2202.10447
    介绍:https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码，个人认为是不必要的；如果觉得有必要，
         可以自行通过a_bias传入。
    """
    
    def __init__(
            self,
            hidden_size=3072, #768
            intermediate_size=6144,# 1536
            attention_key_size=128,
            activation="swish",
            use_bias=False,
            normalization="softmax",# softmax_plus
            attention_scale=True,
            attention_dropout=0.1,
            scaling_factor="n",
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.intermediate_size = intermediate_size
        self.attention_key_size = attention_key_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.scaling_factor = scaling_factor
        
        self.i_dense = nn.Linear(
            hidden_size, 2 * intermediate_size + attention_key_size, bias=self.use_bias
        )
        self.o_dense = nn.Linear(intermediate_size, hidden_size, bias=self.use_bias)
        
        self.q_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)
    
    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    
    def forward(
            self,
            hidden_states,
            x_gcn,
            attention_mask=None,
            sinusoidal_pos=None,
            output_attentions=False,
    ):
        # 投影变换
        x = self.i_dense(hidden_states)
        u, v, z = torch.split(
            self.activation(x),
            [self.intermediate_size, self.intermediate_size, self.attention_key_size],
            dim=-1,
        )
        # print(z.shape) 80 17 128
        #q, k = self.q_scaleoffset(z), self.k_scaleoffset(z)
        #print(q.shape) # B K 128
        # 
        q = x_gcn
        k = self.k_scaleoffset(z)
        
        # 加入RoPE
        q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(
            k, sinusoidal_pos
        )
        
        # Attention
        a = torch.einsum("bmd,bnd->bmn", q, k)
        
        if self.attention_scale:
            a = a / self.attention_key_size ** 0.5
        
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, :]
            a = a.masked_fill(attention_mask == 0, -INF)
            l = attention_mask.sum(-1, keepdim=True)
        else:
            l = x.shape[1]
        
        A = attention_normalize(a, l, dim=-1, method=self.normalization, scaling_factor=self.scaling_factor)
        
        A = F.dropout(A, p=self.attention_dropout, training=self.training)
        
        # 计算输出
        o = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))
        
        outputs = (o, A) if output_attentions else (o,)
        return outputs
    
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
    
    def forward1(self, x, mask):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        #print(avgout.shape) # torch.Size([2, 1, 64, 48])
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out)) # 这里要改成 2 -> 1

        out_prm = self.conv_bn_relu_prm_3_1(x)
        out_prm = self.conv_bn_relu_prm_3_2(out_prm)
        out_prm = self.sigmoid3(out_prm)
        #print(out_prm.shape)
        final_out = (out + out_prm + mask) / 3
        #print(final_out.shape)
        return final_out
    
    def forward2(self, x, mask):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        #print(avgout.shape) # torch.Size([2, 1, 64, 48])
        #out = torch.cat([avgout, maxout], dim=1)
        #out = self.sigmoid(self.conv2d(out)) # 这里要改成 2 -> 1

        out_prm = self.conv_bn_relu_prm_3_1(x)
        out_prm = self.conv_bn_relu_prm_3_2(out_prm)
        out_prm = self.conv_(out_prm)
        #self.conv2d = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=7, stride=1, padding=3)

        #out_prm = self.sigmoid3(out_prm)
        
        final_out = torch.cat([avgout, maxout, out_prm, mask], dim=1)
        final_out = self.sigmoid(self.conv2d(final_out)) # 这里要改成 4 -> 1

        return final_out
    
    # 这个最后试，根据后两个的实验情况。
    def forward3(self, x, mask):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        #print(avgout.shape) # torch.Size([2, 1, 64, 48])
        out = torch.cat([avgout, maxout, mask], dim=1)
        out = self.sigmoid(self.conv2d(out))

        out_prm = self.conv_bn_relu_prm_3_1(x)
        out_prm = self.conv_bn_relu_prm_3_2(out_prm)
        out_prm = self.sigmoid3(out_prm)
        
        final_out = (out + out_prm) / 2

        return final_out
    
    # 原来的。改成self-attention？参数量会比较大
    def forward(self, x, mask):
        b,h,w = x.size(0), x.size(2), x.size(3)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        #print(avgout.shape) # torch.Size([2, 1, 64, 48])
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out) # 这里要改成 2 -> 1

        self.patch_embed()
        out = F.interpolate(out, scale_factor=0.5)
        out = rearrange(out, 'b c h w -> b c (h w)') 
        mask = rearrange(mask, 'b c h w -> b c (h w)') 
        final_out = self.gau(out, mask, None, None, False)
        final_out = final_out.view(b, 1, h, w)  
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        final_out = self.sigmoid(final_out) 

        return final_out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule(channel)

    def forward(self, x, mask):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out, mask) * out
        return out


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
        #self.bn = nn.BatchNorm2d(3, momentum=BN_MOMENTUM)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
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
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.cbam32 = CBAM(channel=32)
        self.cbam64 = CBAM(channel=64)
        self.cbam128 = CBAM(channel=128)
        self.conv31 = nn.Conv2d(32, 31, kernel_size = 1, stride = 1, padding = 0)
        self.conv63 = nn.Conv2d(64, 63, kernel_size = 1, stride = 1, padding = 0)
        self.conv127 = nn.Conv2d(128, 127, kernel_size = 1, stride = 1, padding = 0)
        #self.weight = nn.Parameter(torch.ones(1))
        # pre_stage_channels[0]=33
        # pre_stage_channels[1]=65
        # pre_stage_channels[2]=129
        #pre_stage_channels[2] = np.array([33, 65, 129])
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        #    list(map(lambda x:x+1, pre_stage_channels)), num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        
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
                            nn.ReLU(inplace=True)
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
                            nn.ReLU(inplace=True)
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

    def forward(self, x, mask):
        mask = mask.to(torch.float32).unsqueeze(1)
        #print(mask[0].min())
        #print(mask.shape) torch.Size([32, 1, 256, 192])
        #print(x.shape)torch.Size([32, 3, 256, 192])
        
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
        y_list = self.stage3(x_list)

        '''
        print(y_list[0].shape)
        print(y_list[1].shape)
        print(y_list[2].shape)
        torch.Size([2, 32, 64, 48])
        torch.Size([2, 64, 32, 24])
        torch.Size([2, 128, 16, 12])
        '''
        
        #mask = mask * self.weight
        x1 = F.interpolate(mask, scale_factor=0.25)
        x2 = F.interpolate(x1, scale_factor=0.5)
        x3 = F.interpolate(x2, scale_factor=0.5)
        

        # print(y_list[1][0][0].max())
        y_list[0] = self.cbam32(torch.cat((self.conv31(y_list[0]),x1),1), x3)
        y_list[1] = self.cbam64(torch.cat((self.conv63(y_list[1]),x2),1), x3)
        y_list[2] = self.cbam128(torch.cat((self.conv127(y_list[2]),x3),1), x3)

        x_list = []
        
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x_ = self.final_layer(y_list[0])

        if self.coord_representation == 'heatmap':
            return x_
        elif self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
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
            #pretrained_state_dict.pop('transition3.3.0.0.weight')
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
