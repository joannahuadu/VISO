_base_ = (
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py'
)

num_classes = 1
load_from = "work_dirs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_55.pth"
embedding_path = "tools/embeddings/dota_v1_class_texts_helicopter_embedding.npy"

model = dict(type='SimpleYOLOWorldDetector',
    num_test_classes=num_classes,
    mm_neck=True,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts= num_classes,
    backbone=dict(with_text_model=False),
    bbox_head = dict(
        head_module=dict(num_classes = num_classes),
    )
)

dota_val_dataset = dict(
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None),
    class_text_path='data/texts/dota_v1_class_texts_helicopter.json')

val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader