_base_ = (
    'yolo_world_v2_s_rep_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py'
)

load_from = "work_dirs/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_rep/best_dota_mAP_epoch_51_rep_conv.pth"

dota_val_dataset = dict(
    ann_file='val/annfiles/',
    data_prefix=dict(img_path='val/images/'),
    batch_shapes_cfg=None)
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader