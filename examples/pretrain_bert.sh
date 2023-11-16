#!/bin/bash

source /workspace/megatron/configs/config_local.sh
export PS1="\[\e[1;32m\]\u@\h:\[\e[1;34m\]\w\$\[\e[0m\] "

# export PATH=/workspace/megatron

export CUDA_DEVICE_MAX_CONNECTIONS=1

DATA_PROCESSED_PATH="${DATA_PATH}dev_text_sentence" # .._text_sentence

BERT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.0001 \
    --train-iters 500 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --data-path $DATA_PROCESSED_PATH \
    --split 949,50,1
"
#    --data-impl mmap \ ??? Strangle does not work now, but worked earlier!!!
#    --merge-file $MERGE_FILE \
#    --vocab-file $VOCAB_FILE \

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 100000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --wandb-entity-name timur-galimzyanov \
    --wandb-project-name dev \
"

#    --tensorboard-dir ${LOGGING_PATH} \
#bash -c "trap 'bash -i' DEBUG; socat TCP-LISTEN:12345,reuseaddr EXEC:bash"
export WANDB_DISABLE_GIT=true
export WANDB_BASE_URL="https://jetbrains.wandb.io"
# torchrun /workspace/megatron/pretrain_bert.py \
torchrun pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH
