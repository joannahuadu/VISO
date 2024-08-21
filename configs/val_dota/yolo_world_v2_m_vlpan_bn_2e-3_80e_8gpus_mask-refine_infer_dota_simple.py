_base_ = (
    'yolo_world_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py'
)

load_from = "work_dirs/"
embedding_path = "tools/embeddings/dota_v1_class_texts_all_embedding.npy"

model = dict(type='SimpleYOLOWorldDetector',
    mm_neck=True,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts=_base_.num_classes,
    backbone=dict(with_text_model=False),)

dota_val_dataset = dict(
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None))
val_dataloader = dict(dataset=dota_val_dataset)

test_dataloader = val_dataloader