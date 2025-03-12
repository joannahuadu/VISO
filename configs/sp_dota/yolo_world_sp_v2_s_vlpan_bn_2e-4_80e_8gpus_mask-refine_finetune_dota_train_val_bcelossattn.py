_base_ = (
    'yolo_world_sp_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val.py')

text_model_name = '/public/home/wang_mq22/workplace/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
train_batch_size_per_gpu = 8

_base_.model.bbox_head.is_split_attn = True
_base_.model.neck.reduce_block_cfg=dict(type='TextKnowledgeAttnBlock')
_base_.model.backbone.text_model.model_name=text_model_name
_base_.model.bbox_head.loss_attn = dict(
                        type='BCELoss',
                        reduction='mean')
_base_.model.bbox_head.is_skip_mask = True
_base_.model.bbox_head.loss_attn_weight = 5


optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
