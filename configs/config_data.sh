DATA_ARGS="
    --split 900,50,50 \
    --train-samples 2000 \
    --val-samples-per-run 1000\
    --eval-interval-samples 1000 \
    --eval-iters-samples 1000 \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval-samples 30000 \
    --wandb-entity-name machine-learning-methods-in-software-engineering \
    --wandb-project-name dev
"

# timur-galimzyanov
# Intervals in global bathes
# --eval-interval 100 \
# --eval-iters 10 \