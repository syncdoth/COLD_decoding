#!/bin/bash

## Abductive

python3 cold_decoding.py  \
	--seed 12 \
	--mode abductive_langevin \
	--pretrained_model t5-base \
    --length 10 \
	--max-length 40 \
	--start 0 \
	--end 5 \
    --topk 5 \
	--verbose \
	--input-file "./data/abductive/small_data.json" \
	--output-dir "./outputs/abductive/" \
	--output-filename 'abductive_t5_base.json' \
    --batch-size 8 \
	--wandb \
	--wandb-project cold_decoding_abductive \
	--wandb-runname abductive_t5_base_topk5