#!/usr/bin/env bash

# Define the array of tasks
tasks=('parking_lot_or_garage' 'lake_or_pond' 'crop_field' 'wind_farm' 'storage_tank' 'port' 'stadium' 'swimming_pool' 'park' 'tower')

# Base directory for the JSON files
base_dir="/mnt/data1/workspace/wmq/YOLO-World/data/texts"

# Loop over each task to generate JSON files
for task in "${tasks[@]}"; do
  echo "Generating JSON for task: $task"
  
  # Create the JSON content
  json_content='[["'${task//_/ }'"]]'
  
  # Create the file path
  json_file="${base_dir}/fmow_${task}.json"
  
  # Write the JSON content to the file
  echo $json_content > $json_file

  echo "Created file: $json_file with content: $json_content"
done
