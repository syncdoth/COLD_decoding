#!/bin/bash

## CommonGen

python3 cold_decoding.py \
	--seed 12 \
	--mode lexical_generation \
	--pretrained_model gpt2-large  \
	--init-temp 1 \
    --length 10 \
	--max-length 40 \
	--num-iters 2000 \
	--min-iters 1000 \
	--constraint-weight 0.5 \
    --abductive-c2-weight 0.1 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 5 \
	--lr-nll-portion 0.6 \
    --topk 5 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,500,1000,1500 \
	--large_gs_std 1,0.5,0.1,0.05  \
	--stepsize-ratio 1  \
    --batch-size 8 \
    --repeat-batch 8 \
    --print-every 200 \
    --input-file "./data/commongen/commongen.dev.football.jsonl" \
	--output-dir "./data/commongen/" \
	--selfcond \
	--selfcond_weight 0.1 \
	--expertise "../output_dir/gpt2-large/sense/football-1_04_00__/expertise/expertise.csv" \
	--forcing on_p50 \
	--num_units 50 \
	--top_n 1 \
	--selfcond_mode force

