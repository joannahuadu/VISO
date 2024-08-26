_base_ = (
    '../../val_dota/'
    'yolo_world_v2_l_remoteclip_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [1,1,1]
num_classes = 1
load_from = "work_dirs/yolo_world_sp_v2_l_remoteclip_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_16.pth"
dataset_name = "fmow"
task = "wind_farm"
work_dir = f"work_dirs/example/remoteclip/cloud_{dataset_name}_{task}"
embedding_path = f"tools/embeddings/remoteclip_{dataset_name}_{task}.npy"
class_text_path=f'data/texts/{dataset_name}_{task}.json'
# data_root = f'data/split_fMoW_1024/samples/{task}/'
data_root = f'data/split_fMoW_1024/val/fMoW_val_1000_h30'
img_suffix = 'jpg'
# model settings
_base_.model_test_cfg.score_thr = 0.01
_base_.model.test_cfg.score_thr = 0.01
model = dict(type='SimpleYOLOWorldDetectorSP',
    # box_type='qbox',
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
                #   box_type='qbox',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  num_classes=num_classes,
                                  is_sparse_levels = is_sparse_levels))
    )

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
#     dict(
#         type='mmdet.Pad', size=img_scale,
#         pad_val=dict(img=(114, 114, 114))),
#     # dict(type='YOLOv5KeepRatioResize', scale=img_scale),
#     # dict(
#     #     type='LetterResize',
#     #     scale=img_scale,
#     #     allow_scale_up=False,
#     #     pad_val=dict(img=114)),
#     dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
#     # dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
#     dict(type='LoadText'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                     'scale_factor', 'texts'))
# ]

dota_val_dataset = dict(
    dataset=dict(
        img_suffix=img_suffix,
        data_root=data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        batch_shapes_cfg=None),
    replace_char = "_",
    class_text_path=class_text_path)
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(iou_thrs=0.05)
test_evaluator = val_evaluator

# _base_.model.neck.mask_vis = True # 这个是画特征图和mask的
# default_hooks = dict(
#     visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True, score_thr = 0.000001)) 
custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
    # dict(type='yolo_world.RunnerHook'),
    # dict(type='yolo_world.BatchIdxHook'),
    # dict(type='yolo_world.ClassTextsHook', # 画图时需要知道有哪儿些文本，这个hook提供
    #      text_path=class_text_path
    #      ), 
    dict(
        type='SPHook',
    )
]