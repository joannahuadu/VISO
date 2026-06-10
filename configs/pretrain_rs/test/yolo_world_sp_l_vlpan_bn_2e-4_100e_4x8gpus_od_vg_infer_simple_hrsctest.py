_base_ = (
    '../yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_1280ft_dotaval.py')

num_classes = 1
neck_reduce_num_heads = [1,1,1] #??
is_sparse_levels = [0,0,0]
score_th = 0.9
embedding_path = "/mnt/data1/workspace/wmq/YOLO-World/tools/embeddings/hrsc_ship_texts_ywspl.npy"
# load_from = "/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/best_dota_mAP_epoch_20.pth"
load_from = "/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/epoch_14.pth"
# model settings
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
                                  is_sparse_levels=is_sparse_levels))
    )
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

hrsc_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/data/data/split_ss_hrschbb_1024_500/',
        test_mode=True,
        ann_file='test/annfiles/',
        img_suffix='bmp',
        data_prefix=dict(img_path='test/images/'),
        batch_shapes_cfg=None),
    replace_char = "_",
    class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/hrsc_ship_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=hrsc_test_dataset)

custom_hooks = [
    dict(
        type='SPHook',
    )
]