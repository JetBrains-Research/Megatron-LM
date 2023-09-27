## TODO remake train-samples into train-iters
MODEL_ARGS="
    --encoder-num-layers 6 \
    --encoder-1-num-layers 6 \
    --encoder-2-num-layers 8 \
    --decoder-num-layers 10 \
    --hidden-size 768 \
    --num-attention-heads 8 \
    --kv-channels 64 \
    --ffn-hidden-size 1024 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.0001 \
    --train-samples 600 \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100
"

# TODO May be rename sent -> subtree
# Note, that finally label would be (max-label-length+2) due to BOS and EOS tokens.
DATA_PROC_ARGS="
--max-sent-num 128 \
--max-label-length 5 \
--max-sent-length 16 \
--language java \
--tree-sitter-path $TREE_SITTER_PATH
"

# TODO somewhere I use old definition of max-sent-length = max-sent-length + 2. Fix it.

#    --train-iters 500 \
#    --lr-decay-iters 1000000 \
#    --lr-decay-style linear \
#    --seq-length 128
