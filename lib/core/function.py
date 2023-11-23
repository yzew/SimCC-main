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

from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image

import time
import logging
import os

import numpy as np
np.set_printoptions(threshold=np.inf)

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, flip_back_simdr
from utils.transforms import transform_preds
from utils.vis import save_debug_images
from core.loss import JointsMSELoss, NMTCritierion


logger = logging.getLogger(__name__)
def train_sa_simdr2(config, train_loader, model, criterion,criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y,target, target_weight,target_weight_2, meta,meta_2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y,gcn_x = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        target_weight = target_weight.cuda(non_blocking=True).float()
        target_weight_2 = target_weight_2.cuda(non_blocking=True)


        loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
        if isinstance(gcn_x, list):
            loss2 = criterion2(gcn_x[0], target, target_weight_2)
            for output in gcn_x[1:]:
                loss2 += criterion2(output, target, target_weight_2)
        else:
            output = gcn_x
            loss2 = criterion2(output, target, target_weight_2)
        loss = loss1 + 2*loss2
        # print('loss1:%f loss2:%f' % (loss1, loss2))
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss1 {loss1.avg:.5f} Loss2 ({loss2.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, loss1 = losses1, loss2 = losses2)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def train_sa_simdr3(config, train_loader, model, criterion,criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y,target, target_weight,target_weight_2, meta,meta_2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y,gcn_x = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        target_weight = target_weight.cuda(non_blocking=True).float()
        #target_weight_2 = target_weight_2.cuda(non_blocking=True)


        loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
        if isinstance(gcn_x, list):
            loss2 = criterion2(gcn_x[0], target)
            for output in gcn_x[1:]:
                loss2 += criterion2(output, target)
        else:
            output = gcn_x
            loss2 = criterion2(output, target)
        loss = loss1 + 0.003* loss2
        
        # loss = loss1/loss1.detach() + 1.2 * loss2/(loss2/loss1).detach()
        #loss = loss1/loss1.detach() + loss2/loss2.detach()
        # print('loss1:%f loss2:%f' % (loss1, loss2))
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss1 {loss1.avg:.5f} Loss2 ({loss2.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, loss1 = losses1, loss2 = losses2)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def train_sa_simdr4(config, train_loader, model, criterion,criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)

        target_weight = target_weight.cuda(non_blocking=True).float()

        loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
        output_x = output_x.unsqueeze(2)
        target_x = target_x.unsqueeze(2)
        output_y = output_y.unsqueeze(3)
        target_y = target_y.unsqueeze(3)
        loss2 = criterion2(output_x, output_y, target_x, target_y)
        lossa = loss1
        loss1 = loss1 + 0.025* loss2
        #loss = loss1/loss1.detach() + loss2/loss2.detach()

        # compute gradient and do update step
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss1.item(), input.size(0))
        losses1.update(lossa.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss1 {loss1.avg:.5f} Loss2 ({loss2.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, loss1 = losses1, loss2 = losses2)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def train_sa_simdr(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y = model(input)
        #output_x, output_y,gcn_x = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        target_weight = target_weight.cuda(non_blocking=True).float()


        loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate_sa_simdr2(config, val_loader, val_dataset, model, criterion, criterion2, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target, target_weight, target_weight_2, meta,meta_2) in enumerate(val_loader):
            # compute output
            #print(target_x.shape) # 2 17 96
            output_x, output_y,gcn_x = model(input)
            #print(output_x.shape) # 2 17 384
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_,gcn_x_flipped = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                gcn_x_flipped = flip_back(gcn_x_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()
                gcn_x_flipped = torch.from_numpy(gcn_x_flipped.copy()).cuda()
                
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]
                    gcn_x_flipped[:, :, :, 1:] = \
                        gcn_x_flipped.clone()[:, :, :, 0:-1]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
                gcn_x = (gcn_x + gcn_x_flipped) * 0.5
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()
            target_weight_2 = target_weight_2.cuda(non_blocking=True).float()
            #print(target_weight)
            #print(output_x.shape) # 384
            #print(target_x.shape) # 96
            loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
            loss2 = criterion2(gcn_x, target, target_weight_2)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses1.update(loss1.item(), num_images)
            losses2.update(loss2.item(), num_images)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.div(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.div(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t' \
                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss1=losses1, loss2=losses2)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses1.avg,
                losses2.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator
def validate_sa_simdr3(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y,target, target_weight,target_weight_2, meta, meta_2) in enumerate(val_loader):
            # compute output
            #print(target_x.shape) # 2 17 96
            output_x, output_y, x_ = model(input)
            #print(output_x.shape) # 2 17 384
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_, x_flipped = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()
                
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]
                                                       
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)

            target_weight = target_weight.cuda(non_blocking=True).float()
            #print(target_weight)
            #print(output_x.shape) # 384
            #print(target_x.shape) # 96
            loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses1.update(loss1.item(), num_images)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.div(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.div(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss1=losses1)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses1.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def validate_sa_simdr4(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            #print(target_x.shape) # 2 17 96
            output_x, output_y = model(input)
            #print(output_x.shape) # 2 17 384
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()
                
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]
                                                       
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)

            target_weight = target_weight.cuda(non_blocking=True).float()
            #print(target_weight)
            #print(output_x.shape) # 384
            #print(target_x.shape) # 96
            loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses1.update(loss1.item(), num_images)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.div(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.div(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss1=losses1)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses1.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def validate_sa_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            #print(target_x.shape) # 2 17 96
            output_x, output_y = model(input)
            #print(output_x.shape) # 2 17 384
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()
                
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]
                                                       
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)

            target_weight = target_weight.cuda(non_blocking=True).float()
            #print(target_weight)
            #print(output_x.shape) # 384
            #print(target_x.shape) # 96
            loss1 = criterion(output_x, output_y, target_x, target_y, target_weight)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses1.update(loss1.item(), num_images)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.div(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.div(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss1=losses1)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses1.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def train_simdr(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y = model(input)

        target = target.cuda(non_blocking=True).long()
        target_weight = target_weight.cuda(non_blocking=True).float()


        loss = criterion(output_x, output_y, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            output_x, output_y = model(input) # [b,num_keypoints,logits]
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                                                                                     
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                      

                output_x = (F.softmax(output_x,dim=2) + F.softmax(output_x_flipped,dim=2))*0.5
                output_y = (F.softmax(output_y,dim=2) + F.softmax(output_y_flipped,dim=2))*0.5
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)

            # strategies to determine the confidence of predicted location
            mask = max_val_x < max_val_y
            max_val_x[mask] = max_val_y[mask]
            # max_val_x = (max_val_x + max_val_y)/2
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.div(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.div(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def train_heatmap(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, mask) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input,mask)
        #print(outputs.shape) # torch.Size([32, 17, 64, 48])
        #print(mask.shape) # torch.Size([32, 256, 192])
        #print(input.shape) # torch.Size([32, 3, 256, 192])


        # image_grid = Image.fromarray(mask[0].mul(255).byte().numpy())
        # image_grid.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")

        # # transform to same  by .convert('RGBA')
        # image_1 = Image.fromarray(mask[0].mul(255).byte().numpy()).convert('RGB')# 
        # image_2 = Image.fromarray(input[0].permute(1,2,0).byte().numpy()).convert('RGB')
        # # if 0 : show image_2; if 1 : show image_1
        # image = Image.blend(image_2, image_1, 0.4)
        # image_1.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")
        # image_2.save("run/example2/" + str(np.random.uniform(10000,20000)) + ".jpg")
        # image.save("run/example2/" + str(np.random.uniform(20000,30000)) + ".jpg")


        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_heatmap(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta,mask) in enumerate(val_loader):
            # compute output
            outputs = model(input,mask)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                mask_flipped = mask.flip(2)
                outputs_flipped = model(input_flipped, mask_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )
from scipy.ndimage.morphology import distance_transform_edt

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """
    
    if radius < 0:
        return mask
    
    B, C, H, W = mask.shape
    _edgemap = np.zeros((B, C, H, W))
    # 循环处理每个 B 维度的 mask
    for b in range(B):
        # 取出当前 B 维度上的 mask
        current_mask = mask[b]

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(current_mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        
        edgemap = np.zeros(current_mask.shape[1:])

        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)    
        edgemap = (edgemap > 0).astype(np.uint8)
        
        # 将处理后的结果放回结果张量的对应位置
        _edgemap[b] = edgemap

    return _edgemap

    

def train_heatmap2(config, train_loader, model, criterion,criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, mask) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        # compute output
        outputs, aspp = model(input)
        #print(outputs.shape) # torch.Size([32, 17, 64, 48])
        #print(mask.shape) # torch.Size([32, 256, 192])
        #print(input.shape) # torch.Size([32, 3, 256, 192])

        #print(mask.shape) #torch.Size([2, 256, 192])
        
        mask = mask.unsqueeze(1).cuda()

        #print(aspp)
        aspp = aspp.cpu()
        image = Image.fromarray(aspp[0].mul(255).byte().numpy())
        image2 = Image.blend(input, aspp, 0.4)
        image.save("/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint2/w32_seg_point2/results/" + i + ".jpg")
        image2.save("/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint2/w32_seg_point2/results/" + i + ".jpg")

        
        
        #print(edgemap.shape)
        # image_grid = Image.fromarray(mask[0].mul(255).byte().numpy())
        # image_grid.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")

        # # transform to same  by .convert('RGBA')
        # image_1 = Image.fromarray(mask[0].mul(255).byte().numpy()).convert('RGB')# 
        # image_2 = Image.fromarray(input[0].permute(1,2,0).byte().numpy()).convert('RGB')
        # # if 0 : show image_2; if 1 : show image_1
        # image = Image.blend(image_2, image_1, 0.4)
        # image_1.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")
        # image_2.save("run/example2/" + str(np.random.uniform(10000,20000)) + ".jpg")
        # image.save("run/example2/" + str(np.random.uniform(20000,30000)) + ".jpg")


        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss1 = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss1 = loss1 + criterion(output, target, target_weight)
        else:
            output = outputs
            loss1 = criterion(output, target, target_weight)
        lossa = 400 * loss1
        loss2 = criterion2(aspp, mask)
        loss1 = 400 * loss1 + loss2
        # compute gradient and do update step
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss1.item(), input.size(0))
        losses1.update(lossa.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss1 {loss1.val:.5f} ({loss1.avg:.5f})\t' \
                  'Loss2 {loss2.val:.5f} ({loss2.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

# aaa
def train_heatmap3(config, train_loader, model, criterion,criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, mask) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        # compute output
        outputs,seg_final_out, seg_body_out, seg_edge_out = model(input)
        #print(outputs.shape) # torch.Size([32, 17, 64, 48])
        #print(mask.shape) # torch.Size([32, 256, 192])
        #print(input.shape) # torch.Size([32, 3, 256, 192])

        #print(mask.shape) #torch.Size([2, 256, 192])
        
        mask = mask.unsqueeze(1)#.cuda()
        
        _edgemap = onehot_to_binary_edges(mask, 2, 1) # h, w
        
        edgemap = torch.from_numpy(_edgemap).float().cuda()
        #print(edgemap.shape)
        # image_grid = Image.fromarray(mask[0].mul(255).byte().numpy())
        # image_grid.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")

        # # transform to same  by .convert('RGBA')
        # image_1 = Image.fromarray(mask[0].mul(255).byte().numpy()).convert('RGB')# 
        # image_2 = Image.fromarray(input[0].permute(1,2,0).byte().numpy()).convert('RGB')
        # # if 0 : show image_2; if 1 : show image_1
        # image = Image.blend(image_2, image_1, 0.4)
        # image_1.save("run/example2/" + str(np.random.uniform(1,10000)) + ".jpg")
        # image_2.save("run/example2/" + str(np.random.uniform(10000,20000)) + ".jpg")
        # image.save("run/example2/" + str(np.random.uniform(20000,30000)) + ".jpg")


        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss1 = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss1 += criterion(output, target, target_weight)
        else:
            output = outputs
            loss1 = criterion(output, target, target_weight)
        
        loss2 = criterion2((seg_final_out, seg_body_out, seg_edge_out), (mask,edgemap))
        loss = loss1 + loss2
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Loss1 {loss1.val:.5f} ({loss1.avg:.5f})\t' \
                  'Loss2 {loss2.val:.5f} ({loss2.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def validate_heatmap2(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta, mask) in enumerate(val_loader):
            # compute output
            outputs,aspp = model(input)
            

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, aspp = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def validate_heatmap_miou(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    confusion_matrix = np.zeros((1, 1))
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta, mask) in enumerate(val_loader):
            # compute output
            outputs,aspp = model(input)

            # aspp2 = F.interpolate(input=aspp, size=(256, 192), mode='bilinear', align_corners=True).cpu()
            # aspp2 = aspp2[0].squeeze(0)
            # image = Image.fromarray(aspp2.mul(255).byte().numpy())
            # image.save("/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint2/w32_seg_point2/results/" + str(i) + "_0.jpg")
            
            # input2 = input[0].squeeze(0).permute(1,2,0)
            # #print(input2.shape)
            # image_i = Image.fromarray(input2.byte().numpy())
            # image_i.save("/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint2/w32_seg_point2/results/" + str(i) + "_1.jpg")
            # image2 = Image.blend(image_i, image.convert('RGB'), 0.4)
            # image2.save("/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint2/w32_seg_point2/results/" + str(i) + "_2.jpg")


            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, aspp = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images


            ###############################################################
            size = mask.size()
            aspp = F.interpolate(input=aspp, size=(
                256, 192), mode='bilinear', align_corners=False)
            confusion_matrix += get_confusion_matrix(mask, aspp, size, 1, -1)

            if i % config.PRINT_FREQ == 0:
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()

                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'mIoU {mIoU:.4f}\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, mIoU=mean_IoU, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output, prefix)
                
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1



    return perf_indicator,mean_IoU, IoU_array, pixel_acc, mean_acc

def validate_heatmap3(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta,mask) in enumerate(val_loader):
            # compute output
            outputs,seg_final_out, seg_body_out, seg_edge_out = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, seg_final_out, seg_body_out, seg_edge_out = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def train_heatmap_(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_heatmap_(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
