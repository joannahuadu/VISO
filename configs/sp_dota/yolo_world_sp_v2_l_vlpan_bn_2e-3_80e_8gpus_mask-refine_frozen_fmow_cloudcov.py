_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# hyper-parameters
train_batch_size_per_gpu = 32
val_batch_size_per_gpu = 32
load_from = "/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_67.pth"

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??

persistent_workers = False
find_unused_parameters = True
# model settings
model = dict(
    type='CloudCoverageDetector',
    data_preprocessor=dict(type='YOLOCDetDataPreprocessor'),
    backbone=dict(frozen_stages=4),
    cloud_model=dict(type='CloudCoverageHead',
                     head_module=dict(type='CloudCoverageHeadModule',
                                      in_channels=[256, 512, _base_.last_stage_out_channels],
                                      widen_factor=_base_.widen_factor,
                                      featmap_strides=_base_.strides,
                                      norm_cfg=_base_.norm_cfg,
                                      act_cfg=dict(type='SiLU', inplace=True)),
                     loss_pre=dict(
                        type='mmdet.MSELoss',
                        reduction='mean')),
    neck=dict(type='YOLOWorldPAFPNSP',
            #   reduce_embed_channels=neck_reduce_embed_channels,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock'),
            #   downsample_block_cfg=dict(type='DownSampleConvSP')
              ),
    bbox_head=dict(type='YOLOWorldRotatedHeadSP',
                   ## TODO add configs               
                ))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        _scope_='yolo_world',
        type='LoadAnnotations', with_bbox=True, with_cloud=True),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        _scope_='yolo_world',
        type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        _scope_='yolo_world',
        type='LoadAnnotations', with_bbox=True, with_cloud=True),
    dict(
        _scope_='yolo_world',
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]
fmow_train_dataset = dict(
      _delete_=True,
      _scope_='yolo_world',
      type='fMoWDataset',
      data_root='/mnt/data1/workspace/wmq/YOLO-World/data/fMoW',
      meta_label='cloud_cover',
      pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=_base_.train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yoloc_collate'),
    dataset=fmow_train_dataset)

fmow_val_dataset = dict(
      _delete_=True,
      _scope_='yolo_world',
      type='fMoWDataset',
      test_mode = True,
      data_root='/mnt/data1/workspace/wmq/YOLO-World/data/fMoW',
      meta_label='cloud_cover',
      pipeline=test_pipeline)
val_dataloader = dict(
      batch_size=val_batch_size_per_gpu,
      dataset=fmow_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='CloudMetric', metric='accuracy')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))