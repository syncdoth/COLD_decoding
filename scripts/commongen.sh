#!/bin/bash

## CommonGen

python3 cold_decoding.py \
	--seed 12 \
	--mode lexical_generation \
	--pretrained_model gpt2-xl  \
	--init-temp 1 \
    --length 10 \
	--max-length 40 \
	--num-iters 2000 \
	--min-iters 1000 \
	--constraint-weight 0.5 \
	--sentence_ngram_weight 0.0 \
	--right_context_pred_weight 1.0 \
    --keyword_weight 0.1 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 5 \
	--lr-nll-portion 1.0 \
    --topk 5 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,500,1000,1500 \
	--large_gs_std 1,0.5,0.1,0.05  \
	--stepsize-ratio 1  \
    --batch-size 8 \
    --repeat-batch 1 \
    --print-every 200 \
    --input-file "./data/commongen/commongen.dev.jsonl" \
	--output-dir "./outputs/commongen/" \
	--output-filename 'commongen_base_params.json' \
	--wandb \
	--wandb-project cold_decoding_reproduce \
	--wandb-runname commongen_base_params
	# --grad_clip 5.0

