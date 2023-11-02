DATA_ARGS="
    --split 800,100,100 \
    --train-samples 700000 \
    --eval-interval-samples 50000 \
    --eval-iters-samples 10000 \
"
#    --eval-interval-samples 25000 \
#    --eval-iters-samples 10000 \

OUTPUT_ARGS="
    --log-interval 500 \
    --save-interval-samples 100000 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name megatron-codeformer-LM \
    --dataset-size-file dataset_size.json \
"
# megatron-codeformer dev
# timur-galimzyanov
# Intervals in global bathes
# --eval-interval 100 \
# --eval-iters 10 \