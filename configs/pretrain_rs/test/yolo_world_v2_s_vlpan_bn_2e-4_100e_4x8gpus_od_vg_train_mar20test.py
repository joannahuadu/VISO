_base_ = (
    '../yolo_world_v2_s_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_1280ft_pretrain.py')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='LoadText', prompt_format='Detect the {}'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

mar20_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/data/data/MAR20/airplane/OBB/',
        test_mode=True,
        ann_file='test/annfiles/',
        img_suffix='jpg',
        data_prefix=dict(img_path='test/images/'),
        batch_shapes_cfg=None),
    replace_char = "_",
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/mar20_airplane_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=mar20_test_dataset)

# model = dict(test_cfg=dict(score_thr=0.2))