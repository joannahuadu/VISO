_base_ = (
    '../yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_1280ft_dotaval.py')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadText', prompt_format='Detect the {}'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

dota_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_dota/',
        test_mode=True,
        data_prefix=dict(img_path='test/images/'),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v1_class_texts.json',
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
    outfile_prefix='/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_1280ft_dotaval/trainval/epoch14/dotav1/Task1')
