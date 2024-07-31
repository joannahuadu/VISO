_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]