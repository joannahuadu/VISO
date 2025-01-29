#!/usr/bin/env bash

# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
CHECKPOINT=work_dirs/yolo_world_sp_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_61.pth
# CHECKPOINT=work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_67.pth
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/checkpoints/best_dota_mAP_epoch_25.pth
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_28.pth
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
# CHECKPOINT=work_dirs/yolo_world_sp_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val/best_dota_mAP_epoch_59.pth
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"

# Define thresholds
# score_ths=(0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
# score_ths=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)
score_ths=(0.01)

# Loop over tasks and modes
for score_th in "${score_ths[@]}"; do
    echo "Running th: $score_th"
    
    # Copy the template config file to the actual config file
    # cp $CONFIG_TEMPLATE $CONFIG
    
    # Use sed to replace the task and mode in the config file
    sed -i "s/^score_th = .*$/score_th = $score_th/" $CONFIG

    # Run the test script with the updated CONFIG
    ./tools/dist_test.sh $CONFIG $CHECKPOINT $CUDA_VISIBLE_DEVICES
done