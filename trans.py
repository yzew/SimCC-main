import torch
state_dict = torch.load("/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint21/w32_seg_point21/model_best.pth")#xxx.pth或者xxx.pt就是你想改掉的权重文件
torch.save(state_dict, "/dataset/wh/wh_code/SimCC-main/output/coco3/segpoint21/w32_seg_point21/model_best2.pth", _use_new_zipfile_serialization=False)
