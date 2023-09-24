#!/bin/bash

source /workspace/megatron/configs/config_local.sh
#source configs/config_global.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PS1="\[\e[1;32m\]\u@\h:\[\e[1;34m\]\w\$\[\e[0m\] "
# export PATH=/workspace/megatron
DATA_PROCESSED_PATH="${DATA_PATH}dev_code_sentence" # .._text_sentence

## TODO make separate encoder1 and encoder 2 num layers!
MODEL_ARGS="
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --micro-batch-size 16 \
    --global-batch-size 16 \
    --lr 0.0001 \
    --train-samples 600 \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100
"
#    --train-iters 500 \
#    --lr-decay-iters 1000000 \
#    --lr-decay-style linear \

DATA_PROC_ARGS="
--max-doc-length 1024
--max-sent-num 128
--max-sent-length 18
--max-label-length 7
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
torchrun pretrain_codeformer.py \
    $MODEL_ARGS \
    $DATA_ARGS \
    $DATA_PROC_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH
# codeformer
