_base_ = (
    '../yolo_world_v2_s_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_1280ft_pretrain.py')

val_batch_size_per_gpu = 1

diorrsvg_test_dataset = dict(
    _delete_=True,
    type='VisualGroundingDataset',
    data_root='/mnt/data1/workspace/wmq/YOLO-World/data/refGeo/',
    ann_file='metainfo/',
    datasets=['dior_rsvg'],
    filter_anns=['dior_rsvg_train', 'dior_rsvg_val'],
    load_type='question_id',
    data_prefix=dict(img_path='images/'),
    pipeline=_base_.test_pipeline)

val_dataloader = dict(batch_size=val_batch_size_per_gpu, dataset=diorrsvg_test_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True, type='DIORRSVGMetric', iou_thrs=0.5)
test_evaluator = val_evaluator