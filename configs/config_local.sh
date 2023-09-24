MEGATRON_PATH="/workspace/megatron/"
DATA_PATH="/workspace/dataset/"
MODEL_PATH="/workspace/model_setup"
CHECKPOINT_PATH="/workspace/checkpoints"
LOGGING_PATH="/workspace/logging"

# DATA_FILE="data_dev.jsonl"
DATA_FILE="train_cf.jsonl"
DATA_FILE_PATH="${DATA_PATH}${DATA_FILE}"

VOCAB_FILE="${MODEL_PATH}/bert-large-uncased-vocab.txt"
MERGE_FILE="gpt2-merges.txt"
## TODO remove VOCAB_FILE as requred parameter if HF tokenizer is used