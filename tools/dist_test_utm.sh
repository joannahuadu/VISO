#!/usr/bin/env bash

# CONFIG_TEMPLATE=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_frozen_fmow_utm_template.py
CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_frozen_fmow_utm.py
CHECKPOINT=work_dirs/yolo_world_sp_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_dota_train_val_bcelossattn/best_dota_mAP_epoch_61.pth
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"

# Define tasks and modes
# tasks=('parking_lot_or_garage' 'lake_or_pond' 'crop_field' 'wind_farm' 'storage_tank' 'port' 'stadium' 'swimming_pool' 'park' 'tower')
# modes=('parking_lot_or_garage' 'lake_or_pond' 'crop_field' 'wind_farm' 'storage_tank' 'port' 'stadium' 'swimming_pool' 'park' 'tower')
# tasks=('lake_or_pond')
# modes=('lake_or_pond')
tasks=('bridge' 'harbor' 'ship' 'plane' 'storage_tank' 'swimming_pool')
# tasks=('harbor')
# modes=('road_bridge' 'parking_lot_or_garage' 'shipyard' 'airport' 'storage_tank' 'swimming_pool')
# modes=('port')
# modes=('road_bridge' 'port' 'shipyard' 'airport_hangar' 'storage_tank' 'swimming_pool')
modes=('all')
# Loop over tasks and modes
for task in "${tasks[@]}"; do
  for mode in "${modes[@]}"; do
    echo "Running task: $task, mode: $mode"
    
    # Copy the template config file to the actual config file
    # cp $CONFIG_TEMPLATE $CONFIG
    
    # Use sed to replace the task and mode in the config file
    sed -i "s/^task = .*$/task = \"$task\"/" $CONFIG
    sed -i "s/^mode = .*$/mode = \"$mode\"/" $CONFIG

    # Run the test script with the updated CONFIG
    ./tools/dist_test.sh $CONFIG $CHECKPOINT $CUDA_VISIBLE_DEVICES
  done
done
