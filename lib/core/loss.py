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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
np.set_printoptions(threshold=np.inf)

from PIL import Image
class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=0,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=0, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight#,
            #ignore_index=ignore_label
        )

    def forward(self, score, target):
        #score= torch.tensor([item.cpu().detach().numpy() for item in score]).cuda()
        #score = score.squeeze(0)
        #print(score[0][0])
        #print(target[0])
        ph, pw = score.size(2), score.size(3)
        target = target.squeeze(1)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=False)

        loss = self.criterion(score, target.long())

        return loss



class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=0):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
    
class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        # self.weight = weight
    def forward(self, input, target):
        
        target_h = target.size(2)
        target_w = target.size(3)
        input = F.interpolate(input, size=(target_h, target_w), mode='bilinear')

        n, c, h, w = input.size()
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0


        weight = torch.from_numpy(weight).cuda()
        log_p = log_p.cuda()
        target_t = target_t.float().cuda()
        # print(log_p.dtype)
        # print(target_t.dtype)
        # print(weight.dtype)
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss     

def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    '''
    print(inp.shape)
    print(multihotmask.shape)
    torch.Size([1, 2, 256, 192])
    torch.Size([1, 1, 256, 192])
    '''
    #print(inp.shape)
    #print(multihotmask.shape)
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no sum med version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )


class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """
    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False, ohem=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = False
        self.fp16 = False
        self.ohem = ohem
        self.ohem_loss = OhemCrossEntropy2dTensor(self.ignore_index).cuda()

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    # def onehot2label(self, target):
    #     # a bug here
    #     label = torch.argmax(target[:, :-1, :, :], dim=1).long()
    #     label[target[:, -1, :, :]] = self.ignore_index
    #     return label

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        
        # torch.Size([1, 2, 256, 192])
        # torch.Size([1, 1, 256, 192])
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :, :, :].float().cuda())).sum(1)) * \
                          (1. - mask.float().cuda())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        # add ohem loss for the final stage
        #target = target.unsqueeze(1)#.cuda()
        if self.fp16:
            weights = target[:, :, :, :].sum(1).half()
        else:
            weights = target[:, :, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1
        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss



class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1).cuda()
        valid_mask = target.ne(self.ignore_index).cuda()
        target = target.long() * valid_mask.long().cuda()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes=1, ignore_index=0,mode='train',
                 edge_weight=1, seg_weight=1, seg_body_weight=1, att_weight=1):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = CrossEntropyLoss2d(ignore_index=ignore_index).cuda()# 
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(size_average=True,
                                               ignore_index=ignore_index).cuda()

        self.seg_body_loss = ImgWtLossSoftNLL(classes=classes,
                                     ignore_index=ignore_index,
                                     upper_bound=1.0, ohem=False).cuda()
        self.edge_ohem_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index, min_kept=5000).cuda()

        self.ignore_index = ignore_index
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.seg_body_weight = seg_body_weight


    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0


        weight = torch.from_numpy(weight).cuda()
        log_p = log_p.cuda()
        target_t = target_t.cuda()

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        filler = torch.ones_like(target) * 255
        return self.edge_ohem_loss(input, torch.where(edge.max(1)[0] > 0.8, target, filler))
    def onehot2label(self, target):
        """
        Args:
            target:

        Returns:

        """
        # print(target.shape) # torch.Size([2, 256, 192])
        #target = target.unsqueeze(1)
        # label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        # #print(label.shape)
        # label[target[:, -1, :, :]] = self.ignore_index

        # image_grid = Image.fromarray(target[0].mul(255).byte().numpy())
        # image_grid.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")

        # image_grid2 = Image.fromarray(label[0].byte().numpy())
        # image_grid2.save("run/example2/" + str(np.random.uniform(1,10000)) + "_.jpg")
        label = target.long().squeeze(1)
        #label[target[:, -1, :, :]] = self.ignore_index
        return label
    
    def forward(self, inputs, targets):
        # seg_in\segmask\是label的形式
        # seg_body_in\seg_bord_mask是onehot的形式 0 1
        seg_in, seg_body_in, edge_in = inputs
        seg_bord_mask, edgemask = targets
        '''
        print(seg_bord_mask.shape)
        print(edgemask.shape)
        torch.Size([2, 1, 256, 192])
        torch.Size([2, 1, 256, 192])
        '''
        segmask = self.onehot2label(seg_bord_mask)
        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(seg_in, segmask.cuda())
        losses['seg_body'] = self.seg_body_weight * self.seg_body_loss(seg_body_in, seg_bord_mask)
        losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edge_in, edgemask)
        losses['edge_ohem_loss'] = self.att_weight * self.edge_attention(seg_in, segmask, edge_in)

        return losses


class BoneLoss(nn.Module):
    """Bone length loss.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        use_target_weight (bool): Option to use weighted bone loss.
            Different bone types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """
    def __init__(self, joint_parents):
        super(BoneLoss, self).__init__()
        self.id_i = [15,13,16,14,11,5,6,5,5,6,7,8,1,0,0,1,2,3,4]
        self.id_j = [13,11,14,12,12,11,12,6,7,8,9,10,2,1,2,3,4,5,6]

        # [15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12], [5,6],[5,7],
        # [6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]
    def forward(self, joint_out, joint_gt):
        if len(joint_out.shape) == 4: # (b, n, h, w) heatmap-based featuremap 
            calc_dim = [2, 3]
        elif len(joint_out.shape) == 3:# (b, n, 2) or (b, n, 3) regression-based result
            calc_dim = -1
        
        J = torch.norm(joint_out[:,self.id_i,:] - joint_out[:,self.id_j,:], p=2, dim=calc_dim, keepdim=False)
        Y = torch.norm(joint_gt[:,self.id_i,:] - joint_gt[:,self.id_j,:], p=2, dim=calc_dim, keepdim=False)
        loss = torch.abs(J-Y)
        return loss.mean()

class BoneLoss2(nn.Module):
    """Bone length loss.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        use_target_weight (bool): Option to use weighted bone loss.
            Different bone types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """
    def __init__(self):
        super(BoneLoss2, self).__init__()
        self.id_i = [15, 13, 16, 14, 11, 5,  6,  5, 5, 6, 7, 8,  1, 0, 0, 1, 2, 3, 4]
        self.id_j = [13, 11, 14, 12, 12, 11, 12, 6, 7, 8, 9, 10, 2, 1, 2, 3, 4, 5, 6]

        # [15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12], [5,6],[5,7],
        # [6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]
    def forward(self, joint_out, joint_out2, joint_gt, joint_gt2):
        if len(joint_out.shape) == 4: # (b, n, h, w) heatmap-based featuremap 
            calc_dim = [2, 3]
        elif len(joint_out.shape) == 3:# (b, n, 2) or (b, n, 3) regression-based result
            calc_dim = -1
        
        J = torch.norm(joint_out[:,self.id_i,:] - joint_out[:,self.id_j,:], p=2, dim=calc_dim, keepdim=False)
        Y = torch.norm(joint_gt[:,self.id_i,:] - joint_gt[:,self.id_j,:], p=2, dim=calc_dim, keepdim=False)
        #loss = torch.abs(J-Y).mean()
        J2 = torch.norm(joint_out2[:,self.id_i,:] - joint_out2[:,self.id_j,:], p=2, dim=calc_dim, keepdim=False)
        Y2 = torch.norm(joint_gt2[:,self.id_i,:] - joint_gt2[:,self.id_j,:], p=2, dim=calc_dim, keepdim=False)
        #loss += torch.abs(J2-Y2).mean()
        loss = torch.abs(J + J2 - Y - Y2)
        return loss.mean()
    

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS] # LogSoftmax其实就是对softmax的结果进行log，即Log(Softmax(x))
        self.criterion_ = nn.KLDivLoss(reduction='none') # torch.nn.kldivloss是PyTorch中的Kullback-Leibler散度损失函数
 
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1) 
        return loss

    def forward(self, output_x, output_y, target_x, target_y, target_weight):
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_x_gt = target_x[:,idx].squeeze()
            coord_y_gt = target_y[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()
            loss += (self.criterion(coord_x_pred,coord_x_gt).mul(weight).mean()) 
            loss += (self.criterion(coord_y_pred,coord_y_gt).mul(weight).mean())
        return loss / num_joints 

class NMTNORMCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTNORMCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
 
        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
 
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = torch.mean(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_x, output_y, target, target_weight):
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_gt = target[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()

            loss += self.criterion(coord_x_pred,coord_gt[:,0]).mul(weight).mean()
            loss += self.criterion(coord_y_pred,coord_gt[:,1]).mul(weight).mean()
        return loss / num_joints

class NMTCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
 
        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
 
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = torch.sum(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_x, output_y, target, target_weight):
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_gt = target[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()
            loss += self.criterion(coord_x_pred,coord_gt[:,0]).mul(weight).sum()
            loss += self.criterion(coord_y_pred,coord_gt[:,1]).mul(weight).sum()
        return loss / batch_size

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
    """
            use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
    """
    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
