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
is_sparse_levels = [1,0,0,0]
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
              score_th=0.5,
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