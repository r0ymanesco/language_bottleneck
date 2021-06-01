python -u train.py \
    --mode 'gs' \
    --vocab_size 1024 \
    --n_bits 8 \
    --bits_s 8 \
    --bits_r 3 \
    --n_epochs 250 \
    --temperature 1.0 \
    --batch_size 8 \
    --lr 0.0001 \
    --early_stopping_thr 0.99 \
