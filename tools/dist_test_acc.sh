#!/usr/bin/env bash

# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_61.pth
# CHECKPOINT=work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_67.pth
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/checkpoints/best_dota_mAP_epoch_25.pth
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_28.pth
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_59.pth
###############################update#######################################
###L###
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_dotav2test.py
# CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/best_dota_mAP_epoch_20.pth
###M###
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_dotav2test.py
# CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/best_dota_mAP_epoch_20.pth

# ###diorrsvg###
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_diorrsvgtest.py
# CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/best_dota_mAP_epoch_20.pth
# CUDA_VISIBLE_DEVICES="0"

# ###mar20###
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_mar20test.py
# CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/epoch_14.pth
# CUDA_VISIBLE_DEVICES="1,2,3,6,7,9"

# ###hrsc###
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_hrsctest.py
# CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/epoch_14.pth
# CUDA_VISIBLE_DEVICES="1,2,3,5,6,7"

# ###hrsc###
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_fairv2trainval.py
# CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/epoch_14.pth
# CUDA_VISIBLE_DEVICES="1,2,3,5,6,7"

###vedai###
CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_vedaitest.py
CHECKPOINT=/mnt/data1/workspace/wmq/YOLO-World/work_dirs/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_train_dotaval_bcelossattn/trainval/epoch_14.pth
CUDA_VISIBLE_DEVICES="1,2,3,5,6,7"

# Define is_sparse_levels and score_ths combinations
is_sparse_levels_vals=("1,1,1" "1,1,0" "1,0,0")
score_ths=(0 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9)

# Loop over all combinations of is_sparse_levels and score_th
for is_sparse_levels in "${is_sparse_levels_vals[@]}"; do
    for score_th in "${score_ths[@]}"; do
        echo "Running for is_sparse_levels: $is_sparse_levels with score_th: $score_th"
        
        # Copy the template config file to the actual config file
        # cp $CONFIG_TEMPLATE $CONFIG
        
        # Update the configuration file with the current values of is_sparse_levels and score_th
        sed -i "s/^is_sparse_levels = .*$/is_sparse_levels = [$is_sparse_levels]/" $CONFIG
        sed -i "s/^score_th = .*$/score_th = $score_th/" $CONFIG
        
        # Run the test script with the updated CONFIG
        PORT=29518 ./tools/dist_test.sh $CONFIG $CHECKPOINT $CUDA_VISIBLE_DEVICES
    done
done

