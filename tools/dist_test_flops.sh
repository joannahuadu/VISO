#!/usr/bin/env bash

# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_s_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_l_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_amask_infer_single_task_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/sp_dota/yolo_world_sp_v2_m_vlpan_bn_2e-3_80e_8gpus_mask-refine_infer_dota_simple.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_dotav2test.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_dotav2test_single_task.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_diorrsvgtest.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_m_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_mar20test.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_fairv2test.py
# CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_fairv2trainval.py
CONFIG=/mnt/data1/workspace/wmq/YOLO-World/configs/pretrain_rs/test/yolo_world_sp_l_vlpan_bn_2e-4_100e_4x8gpus_od_vg_infer_simple_dotav2test_single_task.py
CUDA_VISIBLE_DEVICES="1"

# Define thresholds
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
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./tools/analysis_tools/get_flops.py $CONFIG
    done
done