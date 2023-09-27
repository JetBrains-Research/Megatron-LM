#!/bin/bash

source /workspace/megatron/configs/config_local_path.sh
source /workspace/megatron/configs/config_model.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export PATH=/workspace/megatron
DATA_PROCESSED_PATH="${DATA_PATH}train_code_sentence" # .._text_sentence

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_ARGS="
    --data-path $DATA_PROCESSED_PATH \
    --vocab-file $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name dev \
"
# timur-galimzyanov

export WANDB_DISABLE_GIT=true
export WANDB_BASE_URL="https://jetbrains.wandb.io"

export CUDA_VISIBLE_DEVICES=2,4,5,6
torchrun $DISTRIBUTED_ARGS pretrain_codeformer.py \
    $MODEL_ARGS \
    $DATA_ARGS \
    --tensor-model-parallel-size 2 \
    $DATA_PROC_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH
# TODO decide, whether we need model parallelism

