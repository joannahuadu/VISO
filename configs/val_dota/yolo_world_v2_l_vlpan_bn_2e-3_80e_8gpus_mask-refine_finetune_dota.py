_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)


# hyper-parameters
num_classes = 15
num_training_classes = 15
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.05
train_batch_size_per_gpu = 4
# load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
load_from = "weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
# text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
# text_model_name = '/public/home/wang_mq22/workplace/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

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
# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
            # frozen_stages = 4),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
            #   freeze_all = True,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldRotatedHead',
                   head_module=dict(type='YOLOWorldRotatedHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes),
                #    prior_generator=dict(
                #                     _delete_=True,
                #                     type='FakeRotatedAnchorGenerator',
                #                     angle_version=angle_version,
                #                     octave_base_scale=4,
                #                     scales_per_octave=3,
                #                     ratios=[1.0, 0.5, 2.0],
                #                     strides=[8, 16, 32]),
                    # prior_generator=dict(
                    #                 type='RotatedMlvlPointGenerator',
                    #                 angle_version=angle_version),
                    prior_generator=dict(
                        type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
                    # bbox_coder=dict(
                    #                 _delete_=True,
                    #                 type='DeltaXYWHTRBBoxCoder',
                    #                 angle_version=angle_version,
                    #                 norm_factor=None,
                    #                 edge_swap=True,
                    #                 proj_xy=True,
                    #                 target_means=(.0, .0, .0, .0, .0),
                    #                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))),
                    bbox_coder=dict(
                        type='DistanceAnglePointCoder', angle_version=angle_version),
                    # ??
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
        # assigner=dict(type='RotateBatchTaskAlignedAssigner', num_classes=num_training_classes)
        # ??
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

# last_transform = [
#     # Delete gt_masks to avoid more computation
#     dict(type='RemoveDataElement', keys=['gt_masks']),
#     # dict(
#     #     type='mmdet.Albu',
#     #     transforms=_base_.albu_train_transforms,
#     #     bbox_params=dict(
#     #         type='BboxParams',
#     #         format='pascal_voc',
#     #         label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
#     #     keymap={
#     #         'img': 'image',
#     #         'gt_bboxes': 'bboxes'
#     #     }),
#     dict(type='YOLOv5HSVRandomAug'),
#     dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
#     # dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(type='mmdet.RandomFlip',
#          prob=0.75,
#          direction=['horizontal', 'vertical', 'diagonal']),
#     dict(type='mmrotate.RandomRotate',
#          prob=0.5,
#          angle_range=180,
#          rotate_type='mmrotate.Rotate',
#          rect_obj_labels=[9, 11]), 
#     dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
#     dict(type='RegularizeRotatedBox',
#          angle_version=angle_version),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

# mosaic_affine_transform = [
#     dict(
#         type='MultiModalMosaic',
#         img_scale=img_scale,
#         pad_val=114.0,
#         pre_transform=pre_transform),
#     dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         max_aspect_ratio=100.,
#         scaling_ratio_range=(1 - _base_.affine_scale,
#                              1 + _base_.affine_scale),
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2),
#         border_val=(114, 114, 114),
#         min_area_ratio=_base_.min_area_ratio,
#         use_mask_refine=_base_.use_mask2refine)
# ]

# train_pipeline = [
#     *pre_transform,
#     *mosaic_affine_transform,
#     dict(
#         type='YOLOv5MultiModalMixUp',
#         prob=_base_.mixup_prob,
#         pre_transform=[*pre_transform,
#                        *mosaic_affine_transform]),
#     *last_transform[:-1],
#     *text_transform
# ]

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
    dict(type='RegularizeRotatedBox', angle_version=angle_version),
    # dict(type='mmdet.PackDetInputs')
    *text_transform
    
]

# train_pipeline_stage2 = [
#     *_base_.train_pipeline_stage2[:-1],
#     *text_transform
# ]

# train_pipeline_stage2 = [
#     *pre_transform,
#     dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
#     dict(
#         type='mmdet.Pad', size=img_scale,
#         pad_val=dict(img=(114, 114, 114))),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         max_aspect_ratio=_base_.max_aspect_ratio,
#         scaling_ratio_range=(1 - _base_.affine_scale,
#                              1 + _base_.affine_scale),
#         border_val=(114, 114, 114),
#         min_area_ratio=_base_.min_area_ratio,
#         use_mask_refine=_base_.use_mask2refine),
#     *text_transform
# ]

dota_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='data/split_ss_dota/',
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='data/texts/dota_v1_class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dota_train_dataset)
# test_pipeline = [
#     *_base_.test_pipeline[:-1],
#     dict(type='LoadText'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'pad_param', 'texts'))
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    # dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(
    #     type='LetterResize',
    #     scale=img_scale,
    #     allow_scale_up=False,
    #     pad_val=dict(img=114)),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

dota_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='data/split_ss_dota/',
        test_mode=True,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        batch_shapes_cfg=None),
    class_text_path='data/texts/dota_v1_class_texts.json',
    # class_text_path='data/texts/dota_v1_class_prompts.json',
    # class_text_path='data/texts/dota_v1_class_texts_plane.json',
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
    # dict(
    #     type='mmdet.PipelineSwitchHook',
    #     switch_epoch=max_epochs - close_mosaic_epochs,
    #     switch_pipeline=train_pipeline_stage2)
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
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')
# # evaluation settings
# val_evaluator = dict(
#     _delete_=True,
#     type='mmdet.CocoMetric',
#     proposal_nums=(100, 1, 10),
#     ann_file='data/coco/annotations/instances_val2017.json',
#     metric='bbox')

# visualizer = dict(type='mmrotate.RotLocalVisualizer')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),]  # refer to https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='mmrotate.RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
