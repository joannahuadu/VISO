_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_p2.py')

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]


dota_train_dataset = dict(
    dataset=dict(
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'))
)
train_dataloader = dict(dataset=dota_train_dataset)

dota_val_dataset = dict(
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None))
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='mmrotate.DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

