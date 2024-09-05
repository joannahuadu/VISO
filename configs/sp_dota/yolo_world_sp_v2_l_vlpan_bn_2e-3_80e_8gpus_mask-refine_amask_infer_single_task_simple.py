_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [1,1,1]
num_classes = 1
score_th = 0.5
# load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_split/best_dota_mAP_epoch_16.pth"
load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/checkpoints/best_dota_mAP_epoch_25.pth"
embedding_path = "tools/embeddings/dota_v1_class_texts_helicopter_embedding.npy"

# model settings
model = dict(type='SimpleYOLOWorldDetectorSP',
    mm_neck=True,
    num_train_classes=_base_.num_training_classes,
    num_test_classes=num_classes,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts=num_classes,
    backbone=dict(with_text_model=False),
    neck=dict(type='YOLOWorldPAFPNSPInfer',
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConvSPInfer', sp_type="vspconv"),
              is_sparse_levels=is_sparse_levels,
              score_th=score_th,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='TextKnowledgeAttnBlock'),
              is_split_attn=True,),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  num_classes=num_classes,
                                  is_sparse_levels = is_sparse_levels))
    )

dota_val_dataset = dict(
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None),
    class_text_path='data/texts/dota_v1_class_texts_helicopter.json')

val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='SPHook',
    )
]