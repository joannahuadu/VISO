_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [0,1,0]

# model settings
model = dict(
    neck=dict(type='YOLOWorldPAFPNSPInfer',
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConvSPInfer', sp_type="vspconv"),
              is_sparse_levels = is_sparse_levels,
              score_th=0,
            #   reduce_embed_channels=neck_reduce_embed_channels,
              downsample_block_cfg=dict(type='DownSampleConvSPInfer', sp_type="spconv"),
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  is_sparse_levels = is_sparse_levels))
    )

dota_val_dataset = dict(
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None))
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='SPHook',
    )
]