_base_ = (
    './yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_dotav2test_1.py')


# vis

_base_.model_test_cfg.score_thr = 0.15
_base_.model.test_cfg.score_thr = 0.15

_base_.model.neck.mask_vis = True
# 这个是把检测结果画出来的hook
default_hooks = dict(
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook', draw=True)) 
class_text_path='data/texts/dota_v2_class_texts.json'
custom_hooks = [ # 加这3个Hook，才能够在推理的时候把mask画出来
    dict(type='yolo_world.RunnerHook'),
    dict(type='yolo_world.BatchIdxHook'),
    dict(type='yolo_world.ClassTextsHook', # 画图时需要知道有哪儿些文本，这个hook提供
         text_path=class_text_path
         ), 
]

_base_.test_dataloader.dataset.dataset.data_root = 'data/split_ss_dotav2_1024_500/test'
