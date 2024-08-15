_base_ = (
    '../val_dota_yolov5u/'
    'yolov5u_world_n_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val.py')

# hyper-parameters
train_batch_size_per_gpu = 4

# load_from = "weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??

# model settings
model = dict(
    neck=dict(type='YOLOWorldPAFPNSP',
            #   reduce_embed_channels=neck_reduce_embed_channels,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock'),
            #   downsample_block_cfg=dict(type='DownSampleConvSP')
              ),
    bbox_head=dict(type='YOLOWorldHeadSP',
                   ## TODO add configs               
                ))

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
