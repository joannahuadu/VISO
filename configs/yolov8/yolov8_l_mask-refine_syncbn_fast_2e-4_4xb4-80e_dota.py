_base_ = (
    'yolov8_l_mask-refine_syncbn_fast_2e-3_4xb4-80e_dota.py')


base_lr = 2e-4

# Modify optimizer config
_base_.optim_wrapper.optimizer.lr = base_lr
