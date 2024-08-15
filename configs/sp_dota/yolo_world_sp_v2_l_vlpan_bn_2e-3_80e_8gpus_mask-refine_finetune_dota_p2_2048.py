_base_ = (
    '../val_dota/'
    'yolo_world_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_finetune_dota.py')

# hyper-parameters
train_batch_size_per_gpu = 1
# load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
load_from = "/public/home/wang_mq22/workplace/YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
text_model_name = '/public/home/wang_mq22/workplace/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

strides = [4, 8, 16, 32]
in_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_channels = [128, 256, 512, _base_.last_stage_out_channels]
out_indices = [1,2,3,4]
neck_embed_channels = [64, 128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 4, 8, _base_.last_stage_out_channels // 2 // 32]
# neck_reduce_embed_channels = [128, 256, 512, _base_.last_stage_out_channels]
neck_reduce_num_heads= [1,1,1,1]
img_scale = (2048, 2048)

# model settings
model = dict(
    backbone=dict(
        image_model=dict(out_indices=out_indices),
        text_model=dict(model_name=text_model_name)),
    neck=dict(type='YOLOWorldPAFPNSP',
              in_channels=in_channels,
              out_channels=out_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              # reduce_embed_channels=neck_reduce_embed_channels,
              reduce_num_heads=neck_reduce_num_heads,
              reduce_block_cfg=dict(type='KnowledgeAttnBlock'),
            #   downsample_block_cfg=dict(type='DownSampleConvSP')
              ),
    bbox_head=dict(type='YOLOWorldRotatedHeadSP',
                   head_module=dict(featmap_strides=strides, 
                                    in_channels=in_channels),
                   prior_generator=dict(strides=strides)
                   ## TODO add configs
                ))

train_pipeline = [
    *_base_.pre_transform,
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmrotate.RandomRotate',
         prob=0.5,
         angle_range=180,
         rotate_type='mmrotate.Rotate',
         rect_obj_labels=[9, 11]), 
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RegularizeRotatedBox', angle_version=_base_.angle_version),
    *_base_.text_transform
    
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'texts'))
]

dota_train_dataset = dict(
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dota_train_dataset)

dota_val_dataset = dict(
    pipeline=test_pipeline)
val_dataloader = dict(dataset=dota_val_dataset)
test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))