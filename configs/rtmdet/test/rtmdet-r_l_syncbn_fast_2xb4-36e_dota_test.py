_base_ = '../rtmdet-r_l_syncbn_fast_2xb4-36e_dota.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=_base_.img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dataloader = dict(
    _delete_=True,
    batch_size=_base_.val_batch_size_per_gpu,
    num_workers=_base_.val_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        batch_shapes_cfg=None,
        pipeline=test_pipeline))

test_evaluator = dict(
    type='mmrotate.DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='work_dirs/rtmdet-r_l_syncbn_fast_2xb4-36e_dota/20240707_201328/Task1')
