source /workspace/megatron/configs/config_local_path.sh
source /workspace/megatron/configs/config_model.sh
source /workspace/megatron/configs/config_data.sh

DATA_FILE="train.jsonl"
TRAIN_FILE="train.jsonl"
VAL_FILE="val.jsonl"
TEST_FILE="test.jsonl"
#TRAIN_FILE="text_train.jsonl"
#VAL_FILE="text_val.jsonl"
#TEST_FILE="text_test.jsonl"
#DATA_FILE_PATH="${DATA_INPUT2_PATH}/${DATA_FILE}"
DATA_FILE_PATH="${DATA_INPUT_PATH}/${TRAIN_FILE} ${DATA_INPUT_PATH}/${VAL_FILE} ${DATA_INPUT_PATH}/${TEST_FILE}"

python "${MEGATRON_PATH}tools/preprocess_data_codeformer.py" \
       --input $DATA_FILE_PATH \
       --task method_naming \
       --separate-split-files \
       --output-prefix "${DATA_PATH}dev" \
       --processed-folder "${DATA_PATH}/processed_dev" \
       --vocab-file $VOCAB_FILE \
       --tree-sitter-path $TREE_SITTER_PATH \
       --tokenizer-type HFTokenizer \
       --tokenizer-model "Salesforce/codet5p-220m"\
       --workers 96\
       --split-sentences \
       $DATA_PROC_ARGS

# --json-keys code label \
# Imporetant! code key
# language_modeling, method_naming