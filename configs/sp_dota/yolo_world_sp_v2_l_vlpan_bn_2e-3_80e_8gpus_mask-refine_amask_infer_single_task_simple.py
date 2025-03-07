_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [0,0,0]
num_classes = 14

is_visable = True
class_text_path='data/texts/dota_v1_class_texts_without_soccer_ball_field.json'
# embedding_path = "tools/embeddings/mask_vis_texts.npy"
embedding_path = "tools/embeddings/dota_v1_class_texts_without_soccer_ball_field.npy"
# embedding_path = "tools/embeddings/mask_vis_texts_2.npy"
work_dir = 'paper_pic/mask-for-feature'

# load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_split/best_dota_mAP_epoch_16.pth"
load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/checkpoints/best_dota_mAP_epoch_25.pth"
data_root = "paper_pic/dota/one_pic"
# data_root = "paper_pic/dota"
# _base_.model_test_cfg.score_thr = 0.01
# _base_.model.test_cfg = _base_.model_test_cfg

_base_.model_test_cfg.score_thr = 0.34
_base_.model.test_cfg.score_thr = 0.34

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
              score_th=0.7,
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
        data_root = data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        batch_shapes_cfg=None),
    class_text_path=class_text_path)

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
    custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
        dict(type='yolo_world.VisInfoHook',
            text_path=class_text_path
            ), 
        dict(
            type='SPHook',
        )
    ]
