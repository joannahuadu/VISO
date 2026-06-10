#!/usr/bin/env bash

# Define the array of tasks
# tasks=('parking_lot_or_garage' 'lake_or_pond' 'crop_field' 'wind_farm' 'storage_tank' 'port' 'stadium' 'swimming_pool' 'park' 'tower')
tasks=('harbor')
# Loop over each task
for task in "${tasks[@]}"; do
  echo "Processing task: $task"
  
  # Set CUDA_VISIBLE_DEVICES and run the Python script
  CUDA_VISIBLE_DEVICES="0" python generate_text_prompts.py \
    --model openai/clip-vit-base-patch32 \
    --text /mnt/data1/workspace/wmq/YOLO-World/data/texts/fmow_${task}.json \
    --out ./embeddings/clip_fmow_${task}.npy
done
