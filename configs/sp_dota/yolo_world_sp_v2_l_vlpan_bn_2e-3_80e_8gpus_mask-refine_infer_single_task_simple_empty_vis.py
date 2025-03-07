_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [1,1,1]
num_classes = 1
load_from = "work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_67.pth"
dataset_name = "fmow"
task = "plane"
work_dir = f"work_dirs/example/remoteclip/{dataset_name}_{task}_one_sample"
embedding_path = f"tools/embeddings/remoteclip_{dataset_name}_{task}.npy"
class_text_path=f'data/texts/{dataset_name}_{task}.json'

_base_.model_test_cfg.score_thr = 0.05
_base_.model.test_cfg.score_thr = 0.05
# _base_.dota_train_dataset.class_text_path = class_text_path
# _base_.train_dataloader.dataset.class_text_path = class_text_path
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
              score_th=0.01,
            #   reduce_embed_channels=neck_reduce_embed_channels,
            #   downsample_block_cfg=dict(type='DownSampleConvSPInfer', sp_type="spconv"),
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock')),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  num_classes=num_classes,
                                  is_sparse_levels = is_sparse_levels))
    )

dota_val_dataset = dict(
    dataset=dict(
        img_suffix='jpg',
        data_root=f'data/split_fMoW_1024/example/airport/sample',
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        batch_shapes_cfg=None),
    class_text_path=class_text_path)
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

# _base_.model.neck.mask_vis = True # 这个是画特征图和mask的
default_hooks = dict(
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True, score_thr = 0.000001)) 
custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
    dict(type='yolo_world.VisInfoHook',
        text_path=class_text_path
        ), 
    dict(type='yolo_world.ClassTextsHook', # 画图时需要知道有哪儿些文本，这个hook提供
         text_path=class_text_path
         ), 
    dict(
        type='SPHook',
    )
]
