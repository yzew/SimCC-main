#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn71.yaml
#CUDA_VISIBLE_DEVICES=0,1 python tools/test.py --cfg experiments/coco/hrnet/sa_simdr/gaugcn71.yaml