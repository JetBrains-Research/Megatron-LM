MEGATRON_PATH="/home/galimzyanov/Megatron-LM"
DATA_PATH="/home/galimzyanov/data/Megatron/dataset_dev"
MODEL_PATH="/home/galimzyanov/data/Megatron/model_setup"
CHECKPOINT_PATH="/home/galimzyanov/data/Megatron/checkpoints_v01a"
LOGGING_PATH="/home/galimzyanov/data/Megatron/logging"

# DATA_FILE="data_dev.jsonl"
DATA_FILE="data_dev_short.jsonl"
DATA_FILE_PATH="${DATA_PATH}${DATA_FILE}"

VOCAB_FILE="${MODEL_PATH}/bert-large-uncased-vocab.txt"
MERGE_FILE="gpt2-merges.txt"