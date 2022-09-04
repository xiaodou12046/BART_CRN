#!/bin/bash

python train.py \
    --dataset_name lap_acos \
    --bart_lr 3e-5 \
    --lr 1e-5 \
	--batch_size 16 \
    --num_workers 16 \
    --n_epochs 100 \
    --beta 0.001 \
    --alpha 1 \
    --seed 56 \
	--warmup 0.01

