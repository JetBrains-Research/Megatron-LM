DATA_ARGS="
    --split 800,100,100 \
    --train-samples 17000000 \
    --eval-interval-samples 50000 \
    --eval-iters-samples 10000 \
    --skip-test
"
#    --skip-train
#     --log-logits
OUTPUT_ARGS="
    --log-interval 500 \
    --save-interval-samples 100000 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name dev \
    --dataset-size-file dataset_size.json \
    --log-timers-to-tensorboard
"

# megatron-codeformer megatron-codeformer-LM dev
# timur-galimzyanov
# Intervals in global bathes
#    --train-samples 1700000 \
#    --eval-interval-samples 50000 \
#    --eval-iters-samples 10000 \