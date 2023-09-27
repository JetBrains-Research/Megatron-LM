source /workspace/megatron/configs/config_local_path.sh
source /workspace/megatron/configs/config_model.sh

python "${MEGATRON_PATH}tools/preprocess_data_codeformer.py" \
       --input ${DATA_FILE_PATH} \
       --output-prefix "${DATA_PATH}dev" \
       --vocab-file ${VOCAB_FILE} \
       --tokenizer-type HFTokenizer \
       --tokenizer-model "Salesforce/codet5p-220m"\
       --workers 8\
       --split-sentences \
       --json-keys label code \
       $DATA_PROC_ARGS
       #$DATA_ARGS
# my-bert
# --tokenizer-type BertWordPieceLowerCase \