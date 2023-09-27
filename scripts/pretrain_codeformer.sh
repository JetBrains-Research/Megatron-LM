#!/bin/bash

source /workspace/megatron/configs/config_local_path.sh
source /workspace/megatron/configs/config_model.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PS1="\[\e[1;32m\]\u@\h:\[\e[1;34m\]\w\$\[\e[0m\] "
# export PATH=/workspace/megatron
DATA_PROCESSED_PATH="${DATA_PATH}train_code_sentence" # .._text_sentence

DATA_ARGS="
    --data-path $DATA_PROCESSED_PATH \
    --tree-sitter-path $TREE_SITTER_PATH \
    --vocab-file $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name dev
"
# timur-galimzyanov

export WANDB_DISABLE_GIT=true
export WANDB_BASE_URL="https://jetbrains.wandb.io"
export CUDA_VISIBLE_DEVICES=2
torchrun pretrain_codeformer.py \
    $MODEL_ARGS \
    $DATA_ARGS \
    $DATA_PROC_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH
# codeformer

