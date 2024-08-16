_base_ = (
    'yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py'
)
neck_reduce_num_heads= [1,1,1] #??
is_sparse_levels = [0,0,0]

_base_.model.neck.is_sparse_levels = is_sparse_levels
_base_.model.bbox_head.head_module.is_sparse_levels = is_sparse_levels

_base_.model.neck.mask_vis = True
# 这个是把检测结果画出来的hook
default_hooks = dict(
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True)) 
class_text_path='data/texts/dota_v1_class_texts.json'
custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
    dict(type='yolo_world.RunnerHook'),
    dict(type='yolo_world.BatchIdxHook'),
    dict(type='yolo_world.ClassTextsHook', # 画图时需要知道有哪儿些文本，这个hook提供
         text_path=class_text_path
         ), 
]
