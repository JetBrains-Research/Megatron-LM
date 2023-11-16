#!/bin/bash
ls
source configs_jettrain/config_local_path.sh
source configs_jettrain/config_model.sh
source configs_jettrain/config_data.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export PATH=/workspace/megatron
DATA_PROCESSED_PATH="${DATA_PATH}/processed_wikitext/train_text_sentence"

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

DATA_PATHS="
    --data-path $DATA_PROCESSED_PATH \
    --tree-sitter-path $TREE_SITTER_PATH \
    --vocab-file $VOCAB_FILE \
    --tensorboard-dir $LOGGING_PATH
"

export WANDB_DISABLE_GIT=true
export WANDB_BASE_URL="https://jetbrains.wandb.io"
torchrun $DISTRIBUTED_ARGS pretrain_codeformer.py \
    --codeformer\
    --separate-split-files \
    --task language_modeling \
    $DATA_PATHS \
    $MODEL_ARGS \
    $DATA_ARGS \
    $DATA_PROC_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH \
#    --finetune
