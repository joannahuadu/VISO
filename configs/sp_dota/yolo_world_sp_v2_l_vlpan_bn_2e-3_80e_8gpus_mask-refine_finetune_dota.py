_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# hyper-parameters
# ! 在这里改train_batch_size_per_gpu的话，需要同时改train_dataloader里面的才行
# train_batch_size_per_gpu = 4

# load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
load_from = "/home/becool1/wd/YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??

# model settings
model = dict(
    neck=dict(type='YOLOWorldPAFPNSP',
            #   reduce_embed_channels=neck_reduce_embed_channels,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='MaxSigmoidAttnBlockSP')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSP',
                   ## TODO add configs
                   
                ))
    #                 prior_generator=dict(
    #                     type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
    #                 bbox_coder=dict(
    #                     type='DistanceAnglePointCoder', angle_version=angle_version),
    #                 loss_cls=dict(
    #                     _delete_=True,
    #                     type='mmdet.QualityFocalLoss',
    #                     use_sigmoid=True,
    #                     beta=qfl_beta,
    #                     loss_weight=loss_cls_weight),
    #                 loss_bbox=dict(
    #                     _delete_=True,
    #                     type='mmrotate.RotatedIoULoss',
    #                     mode='linear',
    #                     loss_weight=loss_bbox_weight),
    #                 angle_version=angle_version,
    #                 angle_coder=dict(type='mmrotate.PseudoAngleCoder'),
    #                 use_hbbox_loss=False,
    #                 loss_angle=None),
