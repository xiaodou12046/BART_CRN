#!/bin/bash

python train.py \
    --dataset_name rest_acos \
    --bart_lr 3e-5 \
    --lr 1e-5 \
	--batch_size 16 \
    --num_workers 16 \
    --n_epochs 100 \
    --beta 0.001 \
    --alpha 0.1 \
    --seed 24 \
	--warmup 0.1

