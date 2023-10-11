DATA_ARGS="
    --split 800,100,100 \
    --train-samples 300000 \
    --eval-interval-samples 10000 \
    --eval-iters-samples 1000 \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval-samples 10000 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name dev \
    --dataset-size-file dataset_size.json \
"

# timur-galimzyanov
# Intervals in global bathes
# --eval-interval 100 \
# --eval-iters 10 \