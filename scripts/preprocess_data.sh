source /workspace/megatron/configs/config_local.sh

python "${MEGATRON_PATH}tools/preprocess_data.py" \
       --input ${DATA_FILE_PATH} \
       --output-prefix "${DATA_PATH}dev" \
       --vocab-file ${VOCAB_FILE_PATH} \
       --tokenizer-type HFTokenizer \
       --tokenizer-model "Salesforce/codet5p-220m"\
       --workers 8\
       --split-sentences
# my-bert
# --tokenizer-type BertWordPieceLowerCase \