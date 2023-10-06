DATA_ARGS="
    --split 800,100,100 \
    --train-samples 3000 \
    --eval-interval-samples 100 \
    --eval-iters-samples 100 \
"
#    --val-samples-per-run 100\
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