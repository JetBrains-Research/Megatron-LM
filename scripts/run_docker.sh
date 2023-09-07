source configs/config.sh

docker run --gpus all -it \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v $MEGATRON_PATH:/workspace/megatron \
-v $DATA_PATH:/workspace/dataset \
-v $CHECKPOINTS_PATH:/workspace/checkpoints \
-w /workspace/megatron \
nvcr.io/nvidia/pytorch:23.08-py3
# --rm
#
#