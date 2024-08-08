#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
CUDA_VISIBLE_DEVICES=$3

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29516}
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
    --launcher pytorch \
    ${@:4}
