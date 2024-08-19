_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val.py')

train_batch_size_per_gpu = 1

_base_.model.bbox_head.is_split_attn = True
_base_.model.neck.reduce_block_cfg=dict(type='TextKnowledgeAttnBlock')
_base_.model.bbox_head.loss_attn = dict(
                        type='BCELoss',
                        reduction='mean')
_base_.model.bbox_head.is_skip_mask = False


optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
