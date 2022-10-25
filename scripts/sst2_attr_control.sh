#!/bin/bash

## attribute controlled generation
function control_gen() {
	prompt=$1
	exp_name=$2
	class_idx=$3
	cw=$4
	lr=$5
	script="python3 cold_decoding.py \
		--seed 12 \
		--mode prompted_generation \
		--pretrained_model gpt2-xl \
		--init-temp 1 \
		--length 40 \
		--max-length 40 \
		--num-iters 2000 \
		--min-iters 1000 \
		--constraint-weight $cw \
		--attr_control_weight 1.0 \
		--stepsize $lr \
		--noise-iters 1 \
		--win-anneal-iters 1000 \
		--grad_clip 5.0 \
		--lr-nll-portion 1.0 \
		--topk 5 \
		--output-lgt-temp 1 \
		--verbose \
		--straight-through  \
		--large-noise-iters 50,500,1000,1500 \
		--large_gs_std 1,0.5,0.1,0.05  \
		--stepsize-ratio 1  \
		--batch-size 5 \
		--repeat-batch 1 \
		--print-every 200 \
		--output-dir './outputs/attr_control/' \
		--output-filename '${exp_name}.json' \
		--use_attribute_classifer \
		--attribute_classifier_path saved_models/gpt2-xl-sst2-pool_last-seed2022/best.pth \
		--num_attributes 2 \
		--pool_method last \
		--attr_cls_idx $class_idx \
		--prompt '$prompt' \
		--wandb \
		--wandb-runname $exp_name"
	eval $script
}
# debug
# control_gen "The potato" "debug" 1 0 0.1

for cw in 0.1 0.2 0.4 0.6 0.8 0.9 1.0; do
	for lr in 0.1 0.05 0.01; do
		# positive
		control_gen "The potato" "potato-cw$cw-lr$lr-gpt2-xl-ctrl_sst2_pos-pool_last" 1 $cw $lr

		# negative
		control_gen "The potato" "potato-cw$cw-lr$lr-gpt2-xl-ctrl_sst2_neg-pool_last" 0 $cw $lr
	done
done
