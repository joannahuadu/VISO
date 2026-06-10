_base_ = (
    '../yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_1280ft_dotaval.py')

num_classes = 10
neck_reduce_num_heads = [1,1,1] #??
is_sparse_levels = [0,0,0]
score_th = 0
# embedding_path = "/mnt/data1/workspace/wmq/YOLO-World/tools/embeddings/mar20_airplane_texts_ywspl.npy"
load_from = "/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_qiyuan_ft_qiyuanval_bcelossattn/epoch_26.pth"
# model settings
# model_test_cfg = dict(score_thr=0.15)

model = dict(
    # type='SimpleYOLOWorldDetectorSP',
    # mm_neck=True,
    # num_test_classes=num_classes,
    # embedding_path=embedding_path,
    # prompt_dim=_base_.text_channels,
    # num_prompts=num_classes,
    # backbone=dict(with_text_model=False),
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
                                  is_sparse_levels=is_sparse_levels)),
    # test_cfg=model_test_cfg,
    )
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

qiyuan_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/split_ss_qiyuan_1024_500/',
        test_mode=True,
        img_suffix='jpg',
        data_prefix=dict(img_path='test/images/'),
        batch_shapes_cfg=None),
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/qiyuan_class_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=qiyuan_test_dataset)

test_evaluator = dict(
    type='QiyuanMetric',
    format_only=True,
    jsonfile= "/mnt/data1/workspace/wmq/YOLO-World/data/qiyuan2/test/test_info.json",
    merge_patches=True,
    outfile_prefix='/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_qiyuan_ft_qiyuanval_bcelossattn/trainval/epoch26/pred_0529_warm')

custom_hooks = [
    dict(
        type='SPHook',
    )
]