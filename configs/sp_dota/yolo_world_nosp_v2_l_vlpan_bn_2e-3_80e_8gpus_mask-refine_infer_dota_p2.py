_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

strides = [4, 8, 16, 32]
in_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_indices = [1,2,3,4]
neck_embed_channels = [64, 128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 4, 8, _base_.last_stage_out_channels // 2 // 32]

# model settings
model = dict(
    backbone=dict(image_model=dict(out_indices=out_indices)),
    neck=dict(
              in_channels=in_channels,
              out_channels=out_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              downsample_block_cfg=dict(type='DownSampleConvSP')),
    bbox_head=dict(
                  head_module=dict(
                                  featmap_strides=strides,
                                  in_channels=in_channels),
                  prior_generator=dict(strides=strides)
                ))