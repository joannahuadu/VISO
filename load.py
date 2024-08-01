# weights = '/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_trainval/best_dota_mAP_epoch_56.pth'
# import torch
# temp = torch.load(weights)
# torch.save(temp, 'new_model.pth', _use_new_zipfile_serialization=False)

import torch
import torch.nn as nn
path = 'new_model_1.pth'
checkpoint = torch.load(path, map_location='cuda:0')
print(checkpoint['state_dict'])
