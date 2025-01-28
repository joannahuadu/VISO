_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 15
num_training_classes = 37
max_epochs = 20  # Maximum training epochs ## TODO
close_mosaic_epochs = 10
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4 ## TODO
weight_decay = 0.025 ## TODO
train_batch_size_per_gpu = 2
load_from = "/mnt/data1/workspace/wmq/YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
# text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
## TODO
# img_scale = (1280, 1280)
img_scale = (1024, 1024)

angle_version = 'le90'
strides = [8, 16, 32]

dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
loss_dfl_weight = 1.5 / 4
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
            model_name=text_model_name,)),
            # frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldRotatedHead',
                   head_module=dict(type='YOLOWorldRotatedHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
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
                    ## TODO
                    # loss_dfl=dict(
                    #     type='mmdet.DistributionFocalLoss',
                    #     reduction='mean',
                    #     loss_weight=loss_dfl_weight),
                    angle_version=angle_version,
                    angle_coder=dict(type='mmrotate.PseudoAngleCoder'),
                    use_hbbox_loss=False,
                    loss_angle=None),
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_training_classes,
            topk=dsl_topk,
            iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
            # RBboxOverlaps2D doesn't support batch input, use loop instead.
            batch_iou=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
        ),
    test_cfg=model_test_cfg,)

# dataset settings
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value='',
         prompt_format='Detect the {}'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
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
    dict(type='RegularizeRotatedBox', angle_version=angle_version),
    ## TODO
    # dict(
    #     type='YOLOv5RandomAffine',
    #     max_rotate_degree=0.0,
    #     max_shear_degree=0.0,
    #     scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
    #     max_aspect_ratio=_base_.max_aspect_ratio,
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
    #     border_val=(114, 114, 114)),
    ## TODO
    # dict(type='YOLOv5HSVRandomAug'),
    *text_transform,
]

dotav2_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_dotav2_1024_500/',
        ann_file='train/annfiles/', ## TODO
        data_prefix=dict(img_path='train/images/'), ## TODO
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v2_class_texts.json',
    pipeline=train_pipeline)

dior_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=DIOR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/dior/',
        ann_file='all/annfiles/',
        img_suffix='jpg',
        data_prefix=dict(img_path='all/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    replace_char = "",
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dior_class_texts.json',
    pipeline=train_pipeline)


fairv2_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=FAIR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_fairv2_1024_500/',
        ann_file='trainval/annfiles/',
        img_suffix='tif',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/fairv2_class_texts.json',
    pipeline=train_pipeline)

nwpuvhr10_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=FAIR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_nwpuvhr10_1024_500/',
        ann_file='train/annfiles/',
        img_suffix='jpg',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/nwpuvhr10_class_texts.json',
    pipeline=train_pipeline)

ucasaod_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=FAIR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_ucasaod_1024_500/',
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/ucasaod_class_texts.json',
    pipeline=train_pipeline)

rsod_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=FAIR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_rsod_1024_500/',
        ann_file='train/annfiles/',
        img_suffix='jpg',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/rsod_class_texts.json',
    pipeline=train_pipeline)

hrrsd_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=FAIR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_hrrsd_1024_500/',
        ann_file='train/annfiles/',
        img_suffix='jpg',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/hrrsd_class_texts.json',
    pipeline=train_pipeline)

visdrone_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        # metainfo=FAIR_METAINFO,
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_visdrone_1024_500/',
        ann_file='trainval/annfiles/',
        img_suffix='jpg',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/visdrone_class_texts.json',
    pipeline=train_pipeline)

## TODO: RSVG
# mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
#                         data_root='/mnt/data1/workspace/wmq/YOLO-World/data/mixed_grounding/',
#                         ann_file='annotations/final_mixed_train_no_coco.json',
#                         data_prefix=dict(img='gqa/images/'),
#                         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#                         pipeline=train_pipeline)

# flickr_train_dataset = dict(
#     type='YOLOv5MixedGroundingDataset',
#     data_root='/mnt/data1/workspace/wmq/YOLO-World/data/flickr/',
#     ann_file='annotations/final_flickr_separateGT_train.json',
#     data_prefix=dict(img='full_images/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         dotav2_train_dataset,
                                         dior_train_dataset,
                                         fairv2_train_dataset,
                                         nwpuvhr10_train_dataset,
                                         ucasaod_train_dataset,
                                         rsod_train_dataset,
                                         hrrsd_train_dataset,
                                         visdrone_train_dataset
                                     ],
                                     ignore_keys=['classes', 'palette']))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='LoadText', prompt_format='Detect the {}'),
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
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_dotav2_1024_500/',
        test_mode=True,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v2_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='mmrotate.DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     max_keep_ckpts=-1,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    # dict(type='mmdet.PipelineSwitchHook',
    #      switch_epoch=max_epochs - close_mosaic_epochs,
    #      switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=10,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),] 
visualizer = dict(
    type='mmrotate.RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')