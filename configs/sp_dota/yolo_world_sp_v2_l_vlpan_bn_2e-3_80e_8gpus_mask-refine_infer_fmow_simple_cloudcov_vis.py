_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [0,0,0]
num_classes = 1
load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_frozen_fmow_cloudcov/epoch_6.pth"
# load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_frozen_fmow_cloudcov/best_fmow_loss_epoch_67.pth"
embedding_path = "tools/embeddings/remoteclip_fmow_plane.npy"
cov_thr = 17

_base_.model_test_cfg.score_thr = 0.15
_base_.model.test_cfg.score_thr = 0.15

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
                #   box_type='hbox',
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
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
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
      mode='val_example',
      test_mode = True,
      data_root='data/fMoW',
      meta_label='cloud_cover'),
    replace_char = "_",
    class_text_path='data/texts/fmow_plane.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='CloudMetric', metric='accuracy', is_infer=True)
test_evaluator = val_evaluator

# custom_hooks = [
#     dict(
#         type='SPHook',
#     )
# ]

# _base_.model.neck.mask_vis = True # 这个是画特征图和mask的
default_hooks = dict(
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True, score_thr = 0.000001)) 
custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
    dict(type='yolo_world.RunnerHook'),
    dict(type='yolo_world.BatchIdxHook'),
    dict(type='yolo_world.ClassTextsHook', # 画图时需要知道有哪儿些文本，这个hook提供
         text_path='data/texts/fmow_plane.json'
         ), 
    dict(
        type='SPHook',
    )
]