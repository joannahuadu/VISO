_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [1,1,1]

is_visable = True
work_dir = 'paper_pic/mask1-clip'
load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_67.pth"
embedding_path = "tools/embeddings/dota_v1_class_texts_all_embedding.npy"
data_root = "paper_pic/dota"
# model settings
model = dict(type='SimpleYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=_base_.num_training_classes,
    num_test_classes=_base_.num_classes,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts=_base_.num_classes,
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
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  is_sparse_levels = is_sparse_levels))
    )

dota_val_dataset = dict(
    dataset=dict(
        data_root = data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        batch_shapes_cfg=None))
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

if not is_visable:
    custom_hooks = [
        dict(
            type='SPHook',
        )
    ]
else:
    _base_.model.neck.mask_vis = True
    # 这个是把检测结果画出来的hook
    default_hooks = dict(
        visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True, score_thr = 0.000001)) 
    class_text_path='data/texts/dota_v1_class_texts.json'
    custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
        dict(type='yolo_world.VisInfoHook',
            text_path=class_text_path
            ), 
        dict(
            type='SPHook',
        )
    ]
