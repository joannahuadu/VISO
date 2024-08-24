_base_ = (
    '../../val_dota/'
    'yolo_world_v2_s_remoteclip_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [0,0,0]
num_classes = 1
load_from = "work_dirs/yolo_world_sp_v2_s_remoteclip_vlpan_bn_2e-3_80e_8gpus_mask-refine_frozen_fmow_cloudcov/epoch_5.pth"
# load_from = "work_dirs/yolo_world_sp_v2_s_remoteclip_vlpan_bn_2e-3_80e_8gpus_mask-refine_frozen_fmow_cloudcov/best_fmow_loss_epoch_64.pth"
embedding_path = "/mnt/data1/workspace/wmq/YOLO-World/tools/embeddings/remoteclip_fmow_storage_tank.npy"
cov_thr = 17

# model settings
model = dict(type='SimpleYOLOWorldDetectorSP',
    cloud_model=dict(type='CloudCoverageHead',
                     head_module=dict(type='CloudCoverageHeadModule',
                                      in_channels=[256, 512, _base_.last_stage_out_channels],
                                      widen_factor=_base_.widen_factor,
                                      featmap_strides=_base_.strides,
                                      norm_cfg=_base_.norm_cfg,
                                      act_cfg=dict(type='SiLU', inplace=True))),
    with_cloud_model=True,
    cov_thr = cov_thr,
    mm_neck=True,
    num_train_classes=_base_.num_training_classes,
    num_test_classes=num_classes,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts=num_classes,
    backbone=dict(with_text_model=False),
    neck=dict(type='YOLOWorldPAFPNSPInfer',
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConvSPInfer', sp_type="vspconv"),
              is_sparse_levels=is_sparse_levels,
              score_th=0.4,
            #   reduce_embed_channels=neck_reduce_embed_channels,
            #   downsample_block_cfg=dict(type='DownSampleConvSPInfer', sp_type="spconv"),
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  box_type='hbox',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  num_classes=num_classes,
                                  is_sparse_levels = is_sparse_levels))
    )

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        _scope_='yolo_world',
        type='LoadAnnotations', with_bbox=True, with_cloud=True, box_type='hbox'),
    dict(type='LoadText'),
    dict(
        _scope_='yolo_world',
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

dota_val_dataset = dict(
    dataset=dict(
      _delete_=True,
      _scope_='yolo_world',
      type='fMoWDataset',
      mode='storage_tank',
      test_mode = True,
      data_root='/mnt/data1/workspace/wmq/YOLO-World/data/fMoW',
      meta_label='cloud_cover'),
    replace_char = "_",
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/fmow_storage_tank.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(type='OurDOTAMetric', task='task2', iou_thrs=0.05)
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='SPHook',
    )
]