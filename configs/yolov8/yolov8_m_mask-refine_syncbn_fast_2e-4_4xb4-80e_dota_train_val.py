_base_ = (
    'yolov8_m_mask-refine_syncbn_fast_2e-4_4xb4-80e_dota.py'
)

dota_train_dataset = dict(
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'))

train_dataloader = dict(dataset=dota_train_dataset)

dota_val_dataset = dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None)
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader
