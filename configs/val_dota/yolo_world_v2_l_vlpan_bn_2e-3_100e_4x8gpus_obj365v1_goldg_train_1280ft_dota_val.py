_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 15
num_training_classes = 80
max_epochs = 20  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.025
train_batch_size_per_gpu = 4
load_from = "pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth"
# text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
img_scale = (1280, 1280)
# model settings
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
    nms=dict(type='nms_quadri', iou_threshold=0.1),  # NMS type and threshold
    max_per_img=2000)  # Max number of detections of each image
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
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldQBoxHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    test_cfg=model_test_cfg)

# dataset settings
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
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]

train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform
]

obj365v1_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='data/objects365v1/',
        ann_file='annotations/objects365_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=train_pipeline)

mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
                        data_root='data/mixed_grounding/',
                        ann_file='annotations/final_mixed_train_no_coco.json',
                        data_prefix=dict(img='gqa/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=train_pipeline)

flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='data/flickr/',
    ann_file='annotations/final_flickr_separateGT_train.json',
    data_prefix=dict(img='full_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         obj365v1_train_dataset,
                                         flickr_train_dataset, mg_train_dataset
                                     ],
                                     ignore_keys=['classes', 'palette']))

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
    # dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='YOLOv5KeepRatioResize', scale=img_scale),
#     dict(
#         type='LetterResize',
#         scale=img_scale,
#         allow_scale_up=False,
#         pad_val=dict(img=114)),
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(type='LoadText'),
#     dict(type='mmdet.PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                     'scale_factor', 'pad_param', 'texts'))
# ]

dota_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_dota/',
        test_mode=True,
        ann_file='val/annfiles/',
        data_prefix=dict(img='val/images/'),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader

# val_evaluator = dict(_delete_=True, type='DOTAMetric', metric='mAP')
val_evaluator = dict(_delete_=True, type='DOTAMetric', metric='mAP', iou_thrs=0.2, predict_box_type='qbox')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
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
