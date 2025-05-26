# 0.001 
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_mar20.py ./weights/yolo_world_baseline/yolo_world_s
# 0.5410
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_mar20.py ./weights/yolo_world_baseline/yolo_world_m
# 0.6110
CUDA_VISIBLE_DEVICES=2 python tools/test.py ./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_mar20.py ./weights/yolo_world_baseline/yolo_world_l
# 0.5520

# hrsc
CUDA_VISIBLE_DEVICES=7 python tools/test.py ./configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_hrsc.py ./weights/yolo_world_baseline/yolo_world_s
# 0.1090

CUDA_VISIBLE_DEVICES=8 python tools/test.py ./configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_hrsc.py ./weights/yolo_world_baseline/yolo_world_m
# 0.1250

CUDA_VISIBLE_DEVICES=9 python tools/test.py ./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_hrsc.py ./weights/yolo_world_baseline/yolo_world_l
# 0.2090

# vedai
CUDA_VISIBLE_DEVICES=7 python tools/test.py ./configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_vedai.py ./weights/yolo_world_baseline/yolo_world_s
# 0.8

CUDA_VISIBLE_DEVICES=8 python tools/test.py ./configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_vedai.py ./weights/yolo_world_baseline/yolo_world_m
# 1.2

CUDA_VISIBLE_DEVICES=9 python tools/test.py ./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_test_vedai.py ./weights/yolo_world_baseline/yolo_world_l
# 1.4
