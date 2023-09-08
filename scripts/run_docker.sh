source configs/config.sh

docker run --gpus all -it --rm \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v $BASH_HISTORY_PATH:/root/.bash_history \
-v $MEGATRON_PATH:/workspace/megatron \
-v $DATA_PATH:/workspace/dataset \
-v $CHECKPOINTS_PATH:/workspace/checkpoints \
-v $MODEL_PATH:/workspace/model_setup/ \
-w /workspace/megatron \
pytorch
# nvcr.io/nvidia/pytorch:23.08-py3
# --rm
#
#