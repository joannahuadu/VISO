_base_ = (
    '../yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_1280ft_dotaval.py')

num_classes = 1
neck_reduce_num_heads = [1,1,1] #??
is_sparse_levels = [0,0,0]
score_th = 0.9
embedding_path = "/mnt/data1/workspace/wmq/YOLO-World/tools/embeddings/dota_v2_class_texts_harbor_embedding_ywspl.npy"
load_from = "/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/epoch_14.pth"
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

class_text_path = "/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v1_class_texts_harbor.json"

# dota_val_dataset = dict(
#     class_text_path='/mnt/data1/workspace/wmq/YOLO-World/data/texts/dota_v1_class_texts_helicopter.json')
# val_dataloader = dict(dataset=dota_val_dataset)
# test_dataloader = val_dataloader


test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    1024,
                    1024,
                ),
                type='mmdet.Pad'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='mmrotate.ConvertBoxType'),
            dict(prompt_format='Detect the {}', type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),]


dota_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        _scope_='yolo_world',
        type='YOLOv5DOTADataset',
        data_root='/mnt/data1/workspace/wmq/YOLO-World/data/dotav2_harbor/',
        test_mode=True,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        batch_shapes_cfg=None),
    class_text_path=class_text_path,
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dota_test_dataset)

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
] 

# 使用数据集中定义的调色板
visualizer = dict(
    type='mmrotate.RotLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    # bbox_color='random'
)

score_thr = 0.015
_base_.model_test_cfg.score_thr = score_thr
_base_.model.test_cfg.score_thr = score_thr
# 'random'
_base_.model.neck.mask_vis = True # 这个是画特征图和mask的
class_text_path = class_text_path
default_hooks = dict(
    visualization=dict(
        type='mmdet.engine.hooks.DetVisualizationHook', 
        draw=True, 
        score_thr = 0.017, 
        show=False
    )
) 
custom_hooks = [ 
    dict(
        type='yolo_world.hooks.VisInfoHook',
        text_path=class_text_path
    ), 
    dict(
        type='yolo_world.hooks.SPHook'
    ),
    dict(
        type='yolo_world.hooks.GTMaskVisHook',
        num_classes=num_classes
    ),
]

work_dir = 'appendix_work_dirs/dota_harbor'
