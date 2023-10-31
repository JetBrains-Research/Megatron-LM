source /workspace/megatron/configs/config_local_path.sh
source /workspace/megatron/configs/config_model.sh
source /workspace/megatron/configs/config_data.sh

#DATA_FILE="python/train.jsonl"
#TRAIN_FILE="python/train.jsonl"
#VAL_FILE="python/val.jsonl"
#TEST_FILE="python/test.jsonl"
#TRAIN_FILE="python_subset/train.jsonl"
#VAL_FILE="python_subset/val.jsonl"
#TEST_FILE="python_subset/test.jsonl"
TRAIN_FILE="wikitext/wiki-text-103-raw-train.jsonl"
VAL_FILE="wikitext/wiki-text-103-raw-val.jsonl"
TEST_FILE="wikitext/wiki-text-103-raw-test.jsonl"
#DATA_FILE_PATH="${DATA_INPUT2_PATH}/${DATA_FILE}"
DATA_FILE_PATH="${DATA_INPUT_PATH}/${TRAIN_FILE} ${DATA_INPUT_PATH}/${VAL_FILE} ${DATA_INPUT_PATH}/${TEST_FILE}"

python "${MEGATRON_PATH}tools/preprocess_data_codeformer.py" \
       --input $DATA_FILE_PATH \
       --task language_modeling \
       --separate-split-files \
       --output-prefix "${DATA_PATH}" \
       --processed-folder "${DATA_PATH}/processed_wikitext" \
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