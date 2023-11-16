#!/bin/bash
ls
source configs_jettrain/config_local_path.sh
source configs_jettrain/config_model.sh
source configs_jettrain/config_data.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
DATA_PROCESSED_PATH="${DATA_PATH}/processed_wikitext/train_text_sentence"
DATA_PATHS="
    --data-path $DATA_PROCESSED_PATH \
    --tree-sitter-path $TREE_SITTER_PATH \
    --vocab-file $VOCAB_FILE \
    --tensorboard-dir $LOGGING_PATH \
"

export WANDB_DISABLE_GIT=true
export WANDB_BASE_URL="https://jetbrains.wandb.io"
torchrun pretrain_codeformer.py \
    --codeformer\
    --separate-split-files \
    --task language_modeling \
    $DATA_PATHS \
    $MODEL_ARGS \
    $DATA_ARGS \
    $DATA_PROC_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH

#    --load $CHECKPOINT_PATH \
#    --finetune
# method_naming
