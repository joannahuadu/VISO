_base_ = (
    '../val_dota/'
    'yolo_world_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# hyper-parameters
# neck_reduce_embed_channels = [256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1] #??
num_classes = 1
load_from = "work_dirs/yolo_world_sp_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_61.pth"
dataset_name = "fmow"
task = "storage_tank"
mode = "all"
work_dir = f"work_dirs/utm/s/{dataset_name}_{task}"
embedding_path = f"/mnt/data1/workspace/wmq/YOLO-World/tools/embeddings/clip_{dataset_name}_{task}.npy"
class_text_path=f'/mnt/data1/workspace/wmq/YOLO-World/data/texts/{dataset_name}_{task}.json'
utm_path = f'/mnt/data1/workspace/wmq/YOLO-World/tools/utm_prototype/s/{dataset_name}_{mode}_{task}.json'
# model settings
model = dict(type='UTMAttnCollector',
    mm_neck=True,
    num_train_classes=_base_.num_training_classes,
    num_test_classes=num_classes,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts=num_classes,
    backbone=dict(with_text_model=False),
    data_preprocessor=dict(with_utm=True),
    neck=dict(type='YOLOWorldPAFPNUTMSP',
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='TextKnowledgeAttnBlock')))

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
      mode=f'train_{mode}',
      test_mode = True,
      data_root='/mnt/data1/workspace/wmq/YOLO-World/data/fMoW',
      meta_label='utm'),
    replace_char = "_",
    class_text_path=class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='UTMMetric', metric='utm', utm_path=utm_path)
test_evaluator = val_evaluator