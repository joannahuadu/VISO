_base_ = (
    'yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val.py'
)

_base_.model.neck.mask_vis = True
default_hooks = dict(
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True))
custom_hooks = [
    dict(type='yolo_world.VisInfoHook',
        text_path=class_text_path
        ), 
]
