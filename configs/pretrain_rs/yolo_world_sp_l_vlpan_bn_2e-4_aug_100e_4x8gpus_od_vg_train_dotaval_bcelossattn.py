_base_ = (
    'yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn.py')

# Base learning rate for optim_wrapper. Corresponding to 1xb8=8 bs
base_lr = 0.00025  # 0.004 / 16
lr_start_factor = 1.0e-5
max_epochs = 100  # Maximum training epochs
# Change train_pipeline for final 10 epochs (stage 2)
num_epochs_stage2 = 10

img_scale = (1024, 1024)  # width, height
# ratio range for random resize
random_resize_ratio_range = (0.1, 2.0)
# Cached images number in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20
# ratio for random rotate
random_rotate_ratio = 0.5
# label ids for rect objs
rotate_rect_obj_labels = [9, 11]

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 1
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = -1

# =======================Unmodified in most cases==================

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmyolo.Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(
        type='mmrotate.RandomRotate',
        prob=random_rotate_ratio,
        angle_range=180,
        rotate_type='mmrotate.Rotate',
        rect_obj_labels=rotate_rect_obj_labels),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmyolo.YOLOv5MixUp',
        use_cached=True,
        max_cached_images=mixup_max_cached_images),
    dict(type='RandomLoadText',
         num_neg_samples=(_base_.num_classes, _base_.num_classes),
         max_num_samples=_base_.num_training_classes,
         padding_to_max=True,
         padding_value='',
         prompt_format='Detect the {}'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(
        type='mmrotate.RandomRotate',
        prob=random_rotate_ratio,
        angle_range=180,
        rotate_type='mmrotate.Rotate',
        rect_obj_labels=rotate_rect_obj_labels),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomLoadText',
         num_neg_samples=(_base_.num_classes, _base_.num_classes),
         max_num_samples=_base_.num_training_classes,
         padding_to_max=True,
         padding_value='',
         prompt_format='Detect the {}'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
for data in _base_.train_dataloader.dataset.datasets:
    data.pipeline = train_pipeline
    
# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts,  # only keep latest 3 checkpoints
        rule='greater',
        save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2),
    dict(type='EmptyCacheHook'),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_epochs_stage2, val_interval_stage2)])
