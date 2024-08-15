_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota_p2_2048.py')


base_lr = 2e-5

# Modify optimizer config
_base_.optim_wrapper.optimizer.lr = base_lr

