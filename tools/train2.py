# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ----------------------------s--------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------

# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w48_256x192_adam_lr1e-3_split2_sigma4.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_lite.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_superlite.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_lite_gau_ca.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_lite_supergau.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_gau_gcn.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32_gau_gcn_tail.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/superlite.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn.yaml
# python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/w32.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn.yaml
# CUDA_VISIBLE_DEVICES=3 python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn6.yaml


# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w48_256x192_adam_lr1e-3_split2_sigma4.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output/coco/pose_hrnet/w48_256x192_adam_lr1e-3_split2_sigma4/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output/coco/pose_hrnet/w32_256x192_adam_lr1e-3_split2_sigma4/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_lite.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output/coco/pose_hrnet_lite/w32_256x192_lite/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_superlite.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/pose_hrnet_superlite/w32_256x192_superlite/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_lite_gau_ca.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/pose_hrnet_lite_gau_ca/w32_lite_gau_ca/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_lite_supergau.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/pose_hrnet_lite_supergau/w32_lite_supergau/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_gau_gcn.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/pose_hrnet_lite_gau_gcn/w32_gau_gcn/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32_gau_gcn_tail.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/pose_hrnet_lite_gcn_tail/w32_gau_gcn_tail/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/superlite.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/superlite/superlite/model_best.pth TEST.USE_GT_BBOX False
# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/gaugcn3/gaugcn/model_best.pth TEST.USE_GT_BBOX False
# CUDA_VISIBLE_DEVICES=3 python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/gaugcn/gaugcn2/model_best.pth TEST.USE_GT_BBOX False


# conda activate bottom-up_hrnet
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn5.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/gaugcn5/gaugcn5/model_best.pth TEST.USE_GT_BBOX False
# gaugcn5  bs=100?
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn5.yaml

# python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/w32.yaml TEST.MODEL_FILE /dataset/wh/wh_code/SimCC-main/output2/coco/w32_gg/w32/model_best.pth TEST.USE_GT_BBOX False


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, NMTCritierion, NMTNORMCritierion, KLDiscretLoss
from core.function import train_heatmap, train_simdr, train_sa_simdr,train_sa_simdr2
from core.function import validate_heatmap, validate_simdr, validate_sa_simdr
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    # logger.info(get_model_summary(model, dump_input))
    
    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total number of parameters: %d" % pytorch_total_params)
    # device=torch.device("cuda:1" )
    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(device)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if cfg.LOSS.TYPE == 'JointsMSELoss':
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif cfg.LOSS.TYPE == 'NMTCritierion':
        criterion = NMTCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'NMTNORMCritierion':
        criterion = NMTNORMCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'KLDiscretLoss':
        criterion = KLDiscretLoss().cuda()        
        # criterion2 = JointsMSELoss(
        #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        # ).cuda()
    else:
        criterion = L1JointLocationLoss().cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, cfg.DATASET.TRAIN_SET2, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        cfg.MODEL.COORD_REPRESENTATION,
        cfg.MODEL.SIMDR_SPLIT_RATIO
    )
    train_dataset2 = eval('dataset.'+cfg.DATASET.DATASET)(
                        num_classes=cfg.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        cfg.MODEL.COORD_REPRESENTATION,
        cfg.MODEL.SIMDR_SPLIT_RATIO
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)  # ,map_location='cuda:3'
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )        

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        if cfg.MODEL.COORD_REPRESENTATION == 'simdr':
            train_simdr(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
            
            perf_indicator = validate_simdr(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict)   
        elif cfg.MODEL.COORD_REPRESENTATION == 'sa-simdr':
            # train_sa_simdr2(cfg, train_loader, model, criterion, criterion2, optimizer, epoch,
            #   final_output_dir, tb_log_dir, writer_dict)
            train_sa_simdr(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
            
            perf_indicator = validate_sa_simdr(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict)  
        elif cfg.MODEL.COORD_REPRESENTATION == 'heatmap':
            train_heatmap(cfg, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict)

            perf_indicator = validate_heatmap(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )


        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
