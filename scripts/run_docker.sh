source config.sh

docker run --gpus all -it --rm -v MEGATRON_PATH:/workspace/megatron \
-v DATA_PATH:/workspace/dataset \
-v CHECKPOINTS_PATH:/workspace/checkpoints \
-w /workspace/megatron \
nvcr.io/nvidia/pytorch:23.08-py3
