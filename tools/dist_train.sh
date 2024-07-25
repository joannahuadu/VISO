#!/usr/bin/env bash

CONFIG=$1
CUDA_VISIBLE_DEVICES=$2
DEBUG=${3:-0}  # 如果没有传递参数，则将DEBUG设置为默认值0
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${MASTER_PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
GPUS=${#ADDR[@]}
echo "Using $GPUS GPUs"
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:4} \
    --debug $DEBUG
