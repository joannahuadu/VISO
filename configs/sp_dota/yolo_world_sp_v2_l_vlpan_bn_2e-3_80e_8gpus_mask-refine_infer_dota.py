_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??

# model settings
model = dict(
    neck=dict(type='YOLOWorldPAFPNSPInfer',
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConvSPInfer'),
              reduce_embed_channels=neck_reduce_embed_channels,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='MaxSigmoidAttnBlock')),
    )
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
    # train_cfg=dict(
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