#!/bin/bash

## CommonGen

python3 cold_decoding.py \
	--seed 12 \
	--mode lexical_generation \
	--pretrained_model gpt2-xl  \
    --length 10 \
	--max-length 40 \
	--start 0 \
	--end 5 \
    --topk 5 \
	--force_tokens \
	--verbose \
    --repeat-batch 1 \
    --input-file "./data/commongen/commongen.dev.jsonl" \
	--output-dir "./outputs/commongen/" \
	--output-filename 'commongen_disjunctive.json'
