#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
CUDA_VISIBLE_DEVICES=$3
DEBUG=${4:-0}  # 如果没有传递参数，则将DEBUG设置为默认值0

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29519}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
GPUS=${#ADDR[@]}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --debug $DEBUG \
    --launcher pytorch \
    ${@:5}
