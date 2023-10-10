source /workspace/megatron/configs/config_local_path.sh
source /workspace/megatron/configs/config_model.sh
source /workspace/megatron/configs/config_data.sh

# DATA_FILE="train.jsonl"
TRAIN_FILE="train.jsonl"
VAL_FILE="val.jsonl"
TEST_FILE="test.jsonl"
# DATA_FILE_PATH="${DATA_PATH}${DATA_FILE}"
DATA_FILE_PATH="${DATA_PATH}${TRAIN_FILE} ${DATA_PATH}${VAL_FILE} ${DATA_PATH}${TEST_FILE}"

python "${MEGATRON_PATH}tools/preprocess_data_codeformer.py" \
       --input $DATA_FILE_PATH \
       --output-prefix "${DATA_PATH}dev" \
       --processed-folder "processed" \
       --separate-split-files \
       --vocab-file $VOCAB_FILE \
       --tree-sitter-path $TREE_SITTER_PATH \
       --tokenizer-type HFTokenizer \
       --tokenizer-model "Salesforce/codet5p-220m"\
       --workers 8\
       --split-sentences \
       --json-keys label code \
       $DATA_PROC_ARGS
       #$DATA_ARGS
# my-bert
# --tokenizer-type BertWordPieceLowerCase \