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


import math



import os
import logging

import torch
import torch.nn as nn
from einops import rearrange, repeat


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

INF = float("inf")

#############################################   GCN   ##################################################
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def adj_mx_from_skeleton():
    num_joints = 17
    edges = [
        [15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12], [5,6],[5,7],
        [6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]]
    return adj_mx_from_edges(num_joints, edges, sparse=False)

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class _NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=1, bn_layer=True):
        super(_NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        assert self.inter_channels > 0

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        else:
            raise Exception('Error feature dimension.')

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample > 1:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample))

    def forward(self, x):
        batch_size = x.size(0)  # x: (b, c, t, h, w)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)
        phi_x = phi_x.expand(-1, -1, h, -1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class GraphNonLocal(_NonLocalBlock):
    def __init__(self, in_channels, inter_channels=None, sub_sample=1, bn_layer=True):
        super(GraphNonLocal, self).__init__(in_channels, inter_channels=inter_channels, dimension=1,
                                            sub_sample=sub_sample, bn_layer=bn_layer)


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout = None):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.nonlocal_ = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.nonlocal_(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out

class SemGCN(nn.Module):
    #     def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
    def __init__(self, adj, hid_dim, coords_dim1=512,coords_dim2=512, num_layers=2, nodes_group=None, p_dropout=None):
        super(SemGCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim1, hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim2, adj)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out


#############################################   GCN   ##################################################



class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


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
    """简单的仿射变换层(最后一维乘上gamma向量并加上beta向量)
    说明:1、具体操作为最后一维乘上gamma向量并加上beta向量;
         2、如果直接指定scale和offset,那么直接常数缩放和平移;
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
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
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
        q, k = self.q_scaleoffset(z), self.k_scaleoffset(z)
        
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


class GAULayer(nn.Module):
    def __init__(
            self,
            hidden_size=3072,
            intermediate_size=6144,
            attention_key_size=128,
            activation="swish",
            use_bias=False,
            normalization="softmax_plus",
            attention_scale=True,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            eps=1e-12,
            scaling_factor="n",
    ):
        super().__init__()
        self.gau = GatedAttentionUnit(
            hidden_size,
            intermediate_size,
            attention_key_size,
            activation,
            use_bias,
            normalization,
            attention_scale,
            attention_dropout,
            scaling_factor,
        )
        self.norm = Norm(eps=eps)
        self.hidden_dropout = hidden_dropout
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            sinusoidal_pos=None,
            output_attentions=False,
    ):
        # 投影变换
        gau_output = self.gau(
            hidden_states, attention_mask, sinusoidal_pos, output_attentions
        )
        
        # dropout and residual
        o = F.dropout(gau_output[0], p=self.hidden_dropout, training=self.training)
        o = self.norm(hidden_states + o)
        
        outputs = (o,) + gau_output[1:]  # add attentions if we output them
        
        return outputs


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

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

class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g
    
 
    
class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        assert  cfg.MODEL.COORD_REPRESENTATION in ['simdr', 'sa-simdr','sa-simdr2', 'heatmap'], 'only simdr and sa-simdr and heatmap supported for pose_resnet_upfree'
        self.norm = Norm(eps=1e-12)
        #self.transto3072 = nn.Linear(128, 3072)
        #self.ca = CA_Block(channel=17)
        self.adj = adj_mx_from_skeleton()
        self.mlpp = nn.Sequential(
            ScaleNorm(3072),
            nn.Linear(3072, 256, bias=False))
        #self.gcn384 = SemGCN(self.adj, 384)
        #self.gcn512 = SemGCN2(self.adj, 512)

        self.gcn384 = SemGCN(self.adj, 512, 256, 384)
        self.gcn512 = SemGCN(self.adj, 512, 256, 512)
        #self.gcn3072 = SemGCN(self.adj, 3072, 512, 512)
        self.gau = GatedAttentionUnit(
            hidden_size=256,#512
            intermediate_size=512,#1024
            attention_key_size=128,  # s
            activation="swish",
            use_bias=False,
            normalization="softmax_plus", # softmax_plus
            attention_scale=True,
            attention_dropout=0.1,
            scaling_factor="n",
        )

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
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
            self.stage3_cfg, num_channels, multi_scale_output=True)


        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        # head
        # if self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
        #     # 3072 -> 192*2
        #     # 3072 -> 256*2
        #     self.mlp_head_x = nn.Linear(512, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
        #     self.mlp_head_y = nn.Linear(512, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))


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
        y_list = self.stage3(x_list)

        x_ = self.final_layer(y_list[0]) # change final_layer

        if self.coord_representation == 'heatmap':
            return x_
        elif self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr' or self.coord_representation == 'sa-simdr2':
            # x_ = self.ca(x_)
            
            x = rearrange(x_, 'b c h w -> b c (h w)') # 32, 17, 3072 # import einops
            
            # channel attention?
            x = self.mlpp(x)  # -> B, K, hidden(256)
            # x = self.gcn3072(x)# 60M
            pred_x_gau = self.gau(x, None, None, False)

            o_x = F.dropout(pred_x_gau[0], p=0.1, training=self.training)
            o_x = self.norm(x + o_x)
            outputs_x = (o_x,) + pred_x_gau[1:]  # add attentions if we output them
            # an di 0 weidu pinjie tezhengtu 
            outputs_x = torch.cat([fm for fm in outputs_x], dim = 0)
            # outputs_x = torch.as_tensor([item.cpu().detach().numpy() for item in outputs_x], device = torch.device('cuda'))
            outputs_x = torch.squeeze(outputs_x, 0)
            #print(outputs_x.shape)
            

            # pred_x = self.mlp_head_x(outputs_x) # 32, 17, 384 nn.Linear
            pred_x = self.gcn384(outputs_x)
            # pred_y = self.mlp_head_y(outputs_x) # 32,17,512
            pred_y = self.gcn512(outputs_x)

            return pred_x, pred_y, x_

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

