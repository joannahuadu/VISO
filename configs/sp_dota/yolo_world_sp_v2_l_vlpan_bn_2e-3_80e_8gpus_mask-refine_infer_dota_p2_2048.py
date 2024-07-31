_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

strides = [4, 8, 16, 32]
in_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_indices = [1,2,3,4]
neck_embed_channels = [64, 128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 4, 8, _base_.last_stage_out_channels // 2 // 32]
# neck_reduce_embed_channels = [128, 256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1,1]
is_sparse_levels = [0,0,0,0]
img_scale = (2048, 2048)

# model settings
model = dict(
    backbone=dict(image_model=dict(out_indices=out_indices)),
    neck=dict(type='YOLOWorldPAFPNSPInfer',
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConvSPInfer'),
              in_channels=in_channels,
              out_channels=out_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              sp_type="vspconv",
              is_sparse_levels = is_sparse_levels,
              score_th=0.509,
            #   reduce_embed_channels=neck_reduce_embed_channels,
              downsample_block_cfg=dict(type='DownSampleConvSPInfer', sp_type="spconv"),
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  featmap_strides=strides,
                                  in_channels=in_channels,
                                  sp_type="vspconv",
                                  is_sparse_levels = is_sparse_levels),
                  prior_generator=dict(strides=strides)
                ))

train_pipeline = [
    *_base_.pre_transform,
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
    dict(type='RegularizeRotatedBox', angle_version=_base_.angle_version),
    *_base_.text_transform
    
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

dota_train_dataset = dict(
    pipeline=train_pipeline)
train_dataloader = dict(dataset=dota_train_dataset)

dota_val_dataset = dict(
    pipeline=test_pipeline)
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader