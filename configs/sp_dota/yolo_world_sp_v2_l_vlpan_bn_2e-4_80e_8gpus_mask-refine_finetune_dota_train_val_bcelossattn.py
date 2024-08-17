_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val.py')

_base_.model.bbox_head.is_split_attn = True
_base_.model.neck.reduce_block_cfg=dict(type='TextKnowledgeAttnBlock')
_base_.model.bbox_head.loss_attn = dict(
                        type='BCELoss',
                        reduction='mean')
_base_.model.bbox_head.is_skip_mask = True