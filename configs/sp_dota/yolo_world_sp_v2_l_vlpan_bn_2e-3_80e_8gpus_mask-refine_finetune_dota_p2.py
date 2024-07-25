_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# hyper-parameters
train_batch_size_per_gpu = 4
# load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
load_from = "/home/becool1/wd/YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

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
              reduce_block_cfg=dict(type='MaxSigmoidAttnBlockSP')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSP',
                   head_module=dict(featmap_strides=strides, 
                                    in_channels=in_channels),
                   prior_generator=dict(strides=strides)
                   ## TODO add configs
                ))
    # bbox_head=dict(type='YOLOWorldRotatedHead',
    #                head_module=dict(type='YOLOWorldRotatedHeadModule',
    #                                 use_bn_head=True,
    #                                 embed_dims=text_channels,
    #                                 num_classes=num_training_classes),
    #             #    prior_generator=dict(
    #             #                     _delete_=True,
    #             #                     type='FakeRotatedAnchorGenerator',
    #             #                     angle_version=angle_version,
    #             #                     octave_base_scale=4,
    #             #                     scales_per_octave=3,
    #             #                     ratios=[1.0, 0.5, 2.0],
    #             #                     strides=[8, 16, 32]),
    #                 # prior_generator=dict(
    #                 #                 type='RotatedMlvlPointGenerator',
    #                 #                 angle_version=angle_version),
    #                 prior_generator=dict(
    #                     type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
    #                 # bbox_coder=dict(
    #                 #                 _delete_=True,
    #                 #                 type='DeltaXYWHTRBBoxCoder',
    #                 #                 angle_version=angle_version,
    #                 #                 norm_factor=None,
    #                 #                 edge_swap=True,
    #                 #                 proj_xy=True,
    #                 #                 target_means=(.0, .0, .0, .0, .0),
    #                 #                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))),
    #                 bbox_coder=dict(
    #                     type='DistanceAnglePointCoder', angle_version=angle_version),
    #                 # ??
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
    # train_cfg=dict(
    #     # assigner=dict(type='RotateBatchTaskAlignedAssigner', num_classes=num_training_classes)
    #     # ??
    #     assigner=dict(
    #         _delete_=True,
    #         type='BatchDynamicSoftLabelAssigner',
    #         num_classes=num_classes,
    #         topk=dsl_topk,
    #         iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
    #         # RBboxOverlaps2D doesn't support batch input, use loop instead.
    #         batch_iou=False),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False
    #     ),
    # test_cfg=model_test_cfg,)
