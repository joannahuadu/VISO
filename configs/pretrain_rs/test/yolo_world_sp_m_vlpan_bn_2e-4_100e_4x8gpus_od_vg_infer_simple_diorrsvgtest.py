_base_ = (
    '../yolo_world_v2_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train-1280ft_pretrain.py')

val_batch_size_per_gpu = 1
num_classes = 1
neck_reduce_num_heads = [1,1,1] #??
is_sparse_levels = [0,0,0]
score_th = 0.9
embedding_path = "/mnt/data1/workspace/wmq/YOLO-World/tools/embeddings/diorrsvg_texts_all_embedding_ywspm.pth"
load_from = "/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/best_dota_mAP_epoch_20.pth"
# model settings
model = dict(type='SimpleYOLOWorldDetectorSP',
    mm_neck=True,
    num_test_classes=num_classes,
    embedding_path=embedding_path,
    prompt_dim=_base_.text_channels,
    num_prompts=num_classes,
    backbone=dict(with_text_model=False),
    neck=dict(type='YOLOWorldPAFPNSPInfer',
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConvSPInfer', sp_type="vspconv"),
              is_sparse_levels=is_sparse_levels,
              score_th=score_th,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='TextKnowledgeAttnBlock'),
              is_split_attn=True,),
    bbox_head=dict(type='YOLOWorldRotatedHeadSPInfer',
                  head_module=dict(type='YOLOWorldRotatedHeadModuleSPInfer',
                                  sp_type="vspconv",
                                  num_classes=num_classes,
                                  is_sparse_levels=is_sparse_levels))
    )

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

val_evaluator = dict(_delete_=True, type='DIORRSVGMetric', iou_thrs=0.7)
test_evaluator = val_evaluator


custom_hooks = [
    dict(
        type='SPHook',
    )
]