#!/bin/bash

source /workspace/megatron/configs/config_local.sh

# export PATH=/workspace/megatron
# source /workspace/megatron/configs/config.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CHECKPOINT_PATH=/workspace/checkpoints
# VOCAB_FILE=/workspace/dataset/bert-large-cased-vocab.txt
# MERGE_FILE=gpt2-merges.txt
# DATA_PATH=/workspace/dataset/data_dev_proc.jsonl # .._text_sentence
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
    --train-iters 200 \
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
    --data-impl mmap \
    --split 949,50,1
"
#    --merge-file $MERGE_FILE \
#    --vocab-file $VOCAB_FILE \

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

#bash -c "trap 'bash -i' DEBUG; socat TCP-LISTEN:12345,reuseaddr EXEC:bash"
torchrun /workspace/megatron/pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH
