#!/bin/bash

## positive attribute controlled generation

python3 cold_decoding.py \
	--seed 12 \
	--mode prompted_generation \
	--pretrained_model gpt2-xl  \
	--init-temp 1 \
    --length 20 \
	--max-length 20 \
	--num-iters 2000 \
	--min-iters 1000 \
	--constraint-weight 0.5 \
    --attr_control_weight 1.0 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--lr-nll-portion 1.0 \
    --topk 5 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,500,1000,1500 \
	--large_gs_std 1,0.5,0.1,0.05  \
	--stepsize-ratio 1  \
    --batch-size 8 \
    --repeat-batch 10 \
    --print-every 200 \
	--output-dir "./outputs/attr_control/" \
	--output-filename "gpt2-xl-ctrl_sst2_pos-pool_max.json" \
	--use_attribute_classifer \
	--attribute_classifier_path saved_models/gpt2-xl-sst2-basic-seed2022/best.pth \
	--num_attributes 2 \
	--pool_method max \
	--attr_cls_idx 1 \
	--prompt "The food at the restaurant was" \
	--wandb
