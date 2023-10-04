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
    --split 800,100,100 \
    --train-samples 500 \
    --val-samples-per-run 100\
    --eval-interval-samples 100 \
    --eval-iters-samples 100 \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 100 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name dev
"
# timur-galimzyanov
# Intervals in global bathes
# --eval-interval 100 \
# --eval-iters 10 \

export WANDB_DISABLE_GIT=true
export WANDB_BASE_URL="https://jetbrains.wandb.io"
export CUDA_VISIBLE_DEVICES=7
torchrun pretrain_codeformer.py \
    --codeformer\
    $MODEL_ARGS \
    $DATA_ARGS \
    $DATA_PROC_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
#    --encoder-seq-length 512 \
#    --decoder-seq-length 512 \
#    --max-position-embeddings 512 \

# --load $CHECKPOINT_PATH
# codeformer

