_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 15
num_training_classes = 15
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 4

persistent_workers = False

img_scale = (1024, 1024)
angle_version = 'le90' 
strides = [8, 16, 32]

dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0  # beta of QualityFocalLoss

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # Decode rbox with angle, For RTMDet-R, Defaults to True.
    # When set to True, use rbox coder such as DistanceAnglePointCoder
    # When set to False, use hbox coder such as DistancePointBBoxCoder
    # different setting lead to different AP.
    decode_with_angle=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.05,  # Threshold to filter out boxes.
    nms=dict(type='nms_rotated', iou_threshold=0.1),  # NMS type and threshold
    max_per_img=2000)  # Max number of detections of each image

model = dict(
    bbox_head=dict(type='YOLOv8RotatedHead',
                   head_module=dict(type='YOLOv8RotatedHeadModule',
                                    num_classes=num_training_classes),
                    prior_generator=dict(
                        type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
                    bbox_coder=dict(
                        type='DistanceAnglePointCoder', angle_version=angle_version),
                    loss_cls=dict(
                        _delete_=True,
                        type='mmdet.QualityFocalLoss',
                        use_sigmoid=True,
                        beta=qfl_beta,
                        loss_weight=loss_cls_weight),
                    loss_bbox=dict(
                        _delete_=True,
                        type='mmrotate.RotatedIoULoss',
                        mode='linear',
                        loss_weight=loss_bbox_weight),
                    angle_version=angle_version,
                    angle_coder=dict(type='mmrotate.PseudoAngleCoder'),
                    use_hbbox_loss=False,
                    loss_angle=None),
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=dsl_topk,
            iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
            # RBboxOverlaps2D doesn't support batch input, use loop instead.
            batch_iou=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
        ),
    test_cfg=model_test_cfg,)

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
]

final_transform = [
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmrotate.RandomRotate',
         prob=0.5,
         angle_range=180,
         rotate_type='mmrotate.Rotate',
         rect_obj_labels=[9, 11]), 
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmyolo.RegularizeRotatedBox', angle_version=angle_version),
    *final_transform
    
]

dota_train_dataset = dict(
        _delete_=True,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='data/split_ss_dota/',
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None,
        pipeline=train_pipeline)
    
train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dota_train_dataset)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

dota_val_dataset = dict(
        _delete_=True,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='data/split_ss_dota/',
        test_mode=True,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        batch_shapes_cfg=None,
        pipeline=test_pipeline)
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='mmrotate.DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best='auto',
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval= 1 ,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOWv5OptimizerConstructor')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),]  # refer to https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='mmrotate.RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
