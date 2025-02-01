_base_ = (
    'yolo_world_sp_l_vlpan_bn_2e-3_100e_4x8gpus_od_vg_train_dotaval.py')

text_model_name = 'openai/clip-vit-base-patch32'
train_batch_size_per_gpu = 2

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
