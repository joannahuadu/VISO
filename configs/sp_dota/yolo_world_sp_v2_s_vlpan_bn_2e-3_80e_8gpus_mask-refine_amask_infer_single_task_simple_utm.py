_base_ = (
    '../val_dota/'
    'yolo_world_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [0,0,0]
num_classes = 1
load_from = "work_dirs/yolo_world_sp_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_61.pth"
dataset_name = "fmow"
task = "storage_tank"
mode = "all"
work_dir = f"work_dirs/utm/{dataset_name}_{task}"
embedding_path = f"tools/embeddings/clip_{dataset_name}_{task}.npy"
class_text_path=f'data/texts/{dataset_name}_{task}.json'
utm_path = f'tools/utm_prototype/s/{dataset_name}_{mode}_{task}.json'
# model settings
model = dict(type='SimpleYOLOWorldDetectorSP',
    with_utm = False,
    utm_path = utm_path,
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
              score_th=0.2,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='TextKnowledgeAttnBlock'),
              is_split_attn=True,),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  num_classes=num_classes,
                                  is_sparse_levels = is_sparse_levels))
    )

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        _scope_='yolo_world',
        type='LoadAnnotations', with_bbox=True, with_utm=True, box_type='hbox'),
    dict(type='LoadText'),
    dict(
        _scope_='yolo_world',
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts', 'utms'))
]

dota_val_dataset = dict(
    dataset=dict(
      _delete_=True,
      _scope_='yolo_world',
      type='fMoWDataset',
      mode='val_random_2000',
      test_mode = True,
      data_root=f'data/fMoW_utm/{task}/s',
      meta_label='utm'),
    replace_char = "_",
    class_text_path=class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='SPHook',
    )
]