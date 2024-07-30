_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# hyper-parameters
train_batch_size_per_gpu = 4
# load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
load_from = "/public/home/wang_mq22/workplace/YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

strides = [4, 8, 16, 32]
in_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_indices = [1,2,3,4]
neck_embed_channels = [64, 128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 4, 8, _base_.last_stage_out_channels // 2 // 32]
# neck_reduce_embed_channels = [128, 256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1,1]


# model settings
model = dict(
    backbone=dict(image_model=dict(out_indices=out_indices)),
    neck=dict(type='YOLOWorldPAFPNSP',
              in_channels=in_channels,
              out_channels=out_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              # reduce_embed_channels=neck_reduce_embed_channels,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock'),
              downsample_block_cfg=dict(type='DownSampleConvSP')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSP',
                   head_module=dict(featmap_strides=strides, 
                                    in_channels=in_channels),
                   prior_generator=dict(strides=strides)
                   ## TODO add configs
                ))

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)