_base_ = (
    'yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')


base_lr = 2e-4

# Modify optimizer config
_base_.optim_wrapper.optimizer.lr = base_lr

