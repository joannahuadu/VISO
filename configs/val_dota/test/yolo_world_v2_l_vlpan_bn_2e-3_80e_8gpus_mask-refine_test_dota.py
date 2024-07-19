_base_ = (
    '../yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

dota_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5DOTADataset',
        data_root='data/split_ss_dota/',
        test_mode=True,
        data_prefix=dict(img='test/images/'),
        batch_shapes_cfg=None),
    class_text_path='data/texts/dota_v1_class_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dota_test_dataset)

test_evaluator = dict(
    type='mmrotate.DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='work_dirs/yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota/20240707_195659/Task1')
