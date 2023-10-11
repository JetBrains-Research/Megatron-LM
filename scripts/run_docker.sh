source configs/config.sh

docker run --gpus all -it --rm \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
--network=host \
-v $BASH_HISTORY_PATH:/root/.bash_history \
-v $MEGATRON_PATH:/workspace/megatron \
-v $DATA_PATH:/workspace/dataset \
-v $DATA_INPUT_PATH:/workspace/dataset_input \
-v $DATA_INPUT2_PATH:/workspace/dataset_input2 \
-v $CHECKPOINTS_PATH:/workspace/checkpoints \
-v $MODEL_PATH:/workspace/model_setup/ \
-v $LOGGING_PATH:/workspace/logging/ \
-w /workspace/megatron \
megatron-env /bin/bash -c "source ~/.profile; exec bash"
# nvcr.io/nvidia/pytorch:23.08-py3
# --rm
#
#export PS1="\[\033[32m\]\u@\h:\w\$\[\033[0m\] "