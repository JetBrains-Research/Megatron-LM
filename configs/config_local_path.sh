# TODO remake it into config yaml file
MEGATRON_PATH="/workspace/megatron/"
DATA_PATH="/workspace/dataset/"
MODEL_PATH="/workspace/model_setup"
CHECKPOINT_PATH="/workspace/checkpoints"
LOGGING_PATH="/workspace/logging"
TREE_SITTER_PATH="/workspace/megatron/codeformer_utils/vendor"

DATA_FILE="train.jsonl"
DATA_FILE_PATH="${DATA_PATH}${DATA_FILE}"
VOCAB_FILE="${MODEL_PATH}/bert-large-uncased-vocab.txt"
MERGE_FILE="gpt2-merges.txt"