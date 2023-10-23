# NB! Flash attention supports only attn head dim > 128
MODEL_ARGS="
    --encoder-num-layers 8 \
    --encoder-1-num-layers 8 \
    --encoder-2-num-layers 8 \
    --decoder-num-layers 8 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --use-flash-attn \
    --position-embedding-type rope\
    --ffn-hidden-size 2048 \
    --micro-batch-size 40 \
    --global-batch-size 160 \
    --optimizer sgd \
    --lr 0.01 \
    --min-lr 0.01 \
    --lr-warmup-init 0.0033\
    --lr-warmup-fraction .99 \
    --sgd-momentum 0.95 \
    --nesterov \
    --weight-decay 1e-4 \
    --clip-grad 5.0 \
    --fp16
    --vocab-extra-ids 100
"
#     --fp16 \
#    --kv-channels 64 \
#     --epochs 2 \
#    --use-flash-attn \

# Note, that finally label would be (max-label-length+2) due to BOS and EOS tokens.
DATA_PROC_ARGS="
--max-context-length 4096 \
--max-sent-num 384 \
--max-label-length 5 \
--max-sent-length 16 \
--language java \
"
