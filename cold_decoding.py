#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import time

import nltk
import numpy as np
import pandas as pd
import torch
import wandb

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from selfcond.expertise import get_expert_reference
from selfcond.generation import force_units_hooks
from selfcond.models import PytorchTransformersModel
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel, T5ForConditionalGeneration

from attribute_classifier.attribute_classifier_model import DoubleHeadModel
from constraints import (attr_control_constraint, expert_activation_constraint, fluency_constraint,
                         keyword_lexical_constraint, keyword_sg_constraint,
                         right_context_pred_constraint, sentence_ngram_similarity_constraint)
from util import (decode_with_model_topk, freeze_module, geometric_mean_fusion, get_keywords, get_text_from_logits,
                  initialize, lm_score_from_logits, one_hot, post_process, post_sent,
                  rank_and_filter, set_random_seeds, to_device, top_k_filter_3d)

stop_words = set(stopwords.words('english'))


def options():
    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project",
                        type=str,
                        default="COLD Decoding",
                        help='runname for wandb')
    parser.add_argument("--wandb-runname", type=str, help='runname for wandb')
    parser.add_argument("--straight-through", action="store_true")  # TODO: meaning?
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    # experiment
    parser.add_argument("--input-file",
                        type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--output-dir", type=str, default="./data/commongen/")
    parser.add_argument("--output-filename", type=str)
    parser.add_argument("--use-back-model", action='store_true')
    parser.add_argument("--back-model", type=str, default="danyaljj/opengpt2_pytorch_backward")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--repeat-batch",
                        type=int,
                        default=1,
                        help="loading data util ith examples.")
    parser.add_argument("--mode",
                        type=str,
                        default='constrained_langevin',
                        choices=[
                            'lexical_generation', 'counterfactual_langevin', 'abductive_langevin',
                            'grammar', 'prompted_generation'
                        ])
    parser.add_argument("--discretize_method",
                        type=str,
                        default="model_over_cold",
                        choices=[
                            "raw", "cold_over_model", "model_over_cold", "geometric"
                            ],
                        help="* raw: simple topk over COLD logits"
                             "* cold_over_model: gpt prob is used to rank cold logits"
                             "* model_over_cold: cold prob is used to rank gpt prob"
                             "* geometric: geometric fusion from PPLM")
    parser.add_argument("--force_tokens", action="store_true")
    # model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length",
                        type=int,
                        default=15,
                        help="maximum length of optimized logits.")
    parser.add_argument("--max-length",
                        type=int,
                        default=50,
                        help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length",
                        type=int,
                        default=0,
                        help="length of optimization window in sequence.")
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    parser.add_argument("--sentence_ngram_weight",
                        type=float,
                        default=1.0,
                        help="without specific reason, this stays 1 in the original implementation")
    parser.add_argument("--right_context_pred_weight",
                        type=float,
                        default=1.0,
                        help="without specific reason, this stays 1 in the original implementation")
    parser.add_argument("--keyword_weight", type=float, default=0.05)
    parser.add_argument("--abductive-filterx",
                        action="store_true",
                        help="filter out keywords included in x")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0, help="length of prefix.")
    parser.add_argument("--counterfactual-max-ngram", type=int, default=6)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp",
                        type=float,
                        default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp",
                        type=float,
                        default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--rl-output-lgt-temp",
                        type=float,
                        default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument(
        "--init-temp",
        type=float,
        default=0.1,
        help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='random', choices=['random', 'original'])
    parser.add_argument("--fluency-temp", type=float, default=0.001)
    parser.add_argument("--constraint-temp", type=float, default=0.3)
    # lr
    parser.add_argument("--stepsize",
                        type=float,
                        default=0.1,
                        help="learning rate in the backward pass.")
    parser.add_argument("--stepsize-ratio", type=float, default=1, help="")
    parser.add_argument("--stepsize-iters", type=int, default=1000, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters",
                        type=int,
                        default=0,
                        help="record best only after N iterations")
    parser.add_argument("--noise-iters",
                        type=int,
                        default=1,
                        help="add noise at every N iterations")
    parser.add_argument("--win-anneal-iters",
                        type=int,
                        default=-1,
                        help="froze the optimization window after N iters")
    parser.add_argument("--constraint-iters",
                        type=int,
                        default=1000,
                        help="add one more group of constraints from N iters")
    parser.add_argument("--grad_clip",
                        type=float,
                        default=0,
                        help="gradient clipping value. 0 means turn off")
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large_gs_std", type=str, default="1", help="Example: '1,0.1'")

    # selfcond (expert units) params
    parser.add_argument("--selfcond", action="store_true", help="whether to use selfcond")
    parser.add_argument("--selfcond_mode",
                        type=str,
                        default="constraint",
                        choices=["constraint", "force"],
                        help="whether to force expert neurons or use them as a constraint")
    parser.add_argument("--selfcond_weight", type=float, default=0.1)
    parser.add_argument("--expertise", type=str, help="Expertise results as CSV file.")
    parser.add_argument(
        "--metric",
        type=str,
        default="ap",
        help="Metric to use to rank experts for generation.",
    )
    parser.add_argument("--forcing", type=str, default="on_p50", help="Forcing value.")
    parser.add_argument(
        "--num_units",
        type=int,
        default=1,
        help=("Number of units (top experts in terms of --metric) to be intervened on during"
              " generation"),
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=1,
        help=("Which set of top units to use. If set to 1, units from [0, --num-units] are used. "
              "If set to 2, units from [--num-units, 2*--num-units] are used. And so on. "
              "If set to 0, --num-units random units are selected."),
    )
    parser.add_argument(
        "--per_layer",
        action="store_true",
        help="If set, force --num-units per layer at a time.",
    )
    parser.add_argument(
        "--only_last_token",
        action="store_true",
        help="If set, force only last token.",
    )
    # attribute control classifer
    parser.add_argument(
        "--use_attribute_classifer",
        action="store_true",
        help="use attribute classifer constraint. (Controlled generation)",
    )
    parser.add_argument(
        "--attribute_classifier_path",
        type=str,
        help="path to the saved checkpoint file.",
    )
    parser.add_argument("--num_attributes",
                        type=int,
                        default=2,
                        help="number of labels in the attribute classifier")
    parser.add_argument("--attr_control_weight", type=float, default=1.0)
    parser.add_argument("--pool_method",
                        type=str,
                        choices=["last", "mean", "max"],
                        help="how to pool hidden states in attr classifier")
    parser.add_argument("--attr_cls_idx",
                        type=int,
                        help="the index of the desired attribute we want to control.")
    parser.add_argument("--prompt", type=str, help="the prompt to use in prompted_generation case.")
    parser.add_argument("--no_fluency_mask_in_attr_control",
                        action='store_true',
                        help="do not use fluency mask (mask_t) for attribute control constraint")
    # optimize not the logit space, but the hidden state space
    parser.add_argument("--optimize_hidden_states",
                        action='store_true',
                        help="optimize not the logit space, but the hidden state space")
    # scale grad
    # https://github.com/shawnlimn/ScaleGrad/blob/b2685f9c8680e731316ca6149c1171fda7af5ead/custom/gpt2/run_gpt2.py
    parser.add_argument("--sg_gamma", type=float, default=1.)
    # geometric mean fusion
    parser.add_argument("--geometric_gamma", type=float, default=0.8)
    args = parser.parse_args()
    return args


def decode(model,
           tokenizer,
           args,
           prompt=None,
           sent_constraint=None,
           keyword_constraint=None,
           constraint_functions=(None,),
           device='cpu',
           model_back=None,
           expertise=None,
           value=None,
           metric=None,
           num_units=1,
           top_n=1,
           use_layers=None,
           only_last_token=False):
    """
    prompt: left context   (prompt in lexical task)
    sent_constraint: optimization target  (original ending in counterfactual task)
    keyword_constraint: (constraint set in lexical constrained task)
    constraint_functions: list of function names to use as constraint.
        currently supports ('sentence_ngram', 'right_context_pred', 'keyword', 'attr_control')
    """
    if "selfcond" in args.mode:
        df, expert_per_layer = get_expert_reference(
            expertise,
            value,
            metric,
            num_units=num_units,
            top_n=top_n,
            use_layers=use_layers,
        )
        if args.selfcond_mode == "force":
            model, df_force = force_units_hooks(
                model=model,
                expertise=expertise,
                value=value,
                metric=metric,
                num_units=num_units,
                top_n=top_n,
                use_layers=use_layers,
                only_last_token=only_last_token,
            )

        model_wrapper = model
        model = model_wrapper.module

    model.eval()

    prompt = "<|endoftext|>" if prompt is None else prompt
    x_encoded = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    x_encoded = x_encoded.unsqueeze(0).repeat(args.batch_size, 1)  # [B, T]
    x_onehot = one_hot(x_encoded, dimension=tokenizer.vocab_size)  # [B, T, V]

    if sent_constraint is not None:
        # delete the "." token we appended before
        z_encoded = torch.tensor(tokenizer.encode(sent_constraint)[1:], dtype=torch.long)
        z_encoded = z_encoded.unsqueeze(0).repeat(args.batch_size, 1)  # [B, T]
        z_onehot = one_hot(z_encoded, dimension=tokenizer.vocab_size)  # [B, T, V]
    else:
        z_encoded = z_onehot = None

    length = args.length
    if length <= 0:
        assert z_encoded is not None, "length <=0 requires `sent_constraint` not to be None."
        length = z_encoded.shape[1] - length

    # obtain z_mask
    if keyword_constraint is not None:
        keywords_encoded = torch.tensor(tokenizer.encode(keyword_constraint), dtype=torch.long)
        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[keywords_encoded] = 1.
        z_mask = torch.tensor(z_mask)

        # [B, T, V]
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)
        # [B, K]
        keywords_encoded = keywords_encoded.unsqueeze(0).repeat(args.batch_size, 1)
    elif sent_constraint is not None:
        # NOTE: if no lexical (~keyword based) constraints, obtain keyword from main constraint
        keywords_encoded = None
        z_words = word_tokenize(sent_constraint[2:])  # delete the ". " token we appended before
        z_nonstop_words = [
            w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()
        ]
        z_nonstop_words += [z_words[0]]  # add the first token
        z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
        z_nonstop = tokenizer.encode(z_nonstop_words)
        print('|' + z_nonstop_words + '|')

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[z_nonstop] = 1.
        z_mask = torch.tensor(z_mask)
        # [B, T, V]
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)
    else:
        keywords_encoded = None
        z_mask = None

    # device management
    (x_encoded, x_onehot, z_encoded, z_onehot, z_mask,
     keywords_encoded) = to_device(device, x_encoded, x_onehot, z_encoded, z_onehot, z_mask,
                                   keywords_encoded)

    if args.verbose:
        print(f"prompt:\t|{prompt}|\n"
              f"main constraint:\t|{sent_constraint}|\n"
              f"keyword constraint:\t|{keyword_constraint}|\n"
              f"length:\t{length}")

    # init logits distribution
    if args.init_mode == 'random':
        init_logits, init_states = initialize(model,
                                              x_encoded,
                                              length,
                                              args.init_temp,
                                              device,
                                              return_hidden_states=args.optimize_hidden_states)
    elif args.init_mode == 'original':
        assert z_onehot is not None, ("--init-mode original means to use original"
                                      " reference; sent_constraint required.")
        init_logits = z_onehot / 0.1
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat([
                init_logits,
                torch.zeros([args.batch_size, length - init_logits.shape[1], tokenizer.vocab_size],
                            device=device)
            ],
                                    dim=1)
    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    for text_i in text:
        print(f"[initial]: {text_i}")

    if args.wandb:
        if not args.wandb_runname:
            args.wandb_runname = f'{args.mode}-{round(time.time() * 1000)}'
        experiment = wandb.init(project=args.wandb_project, name=args.wandb_runname, config=args)
        text_table = wandb.Table(columns=["step", "prompt", "keywords", "generation", "ppl"])

    assert args.prefix_length <= 0  # Otherwise not compatible with batch mode

    if args.prefix_length > 0:
        prefix_logits = torch.nn.Parameter(
            torch.rand(x_onehot.shape[0],
                       args.prefix_length,
                       x_onehot.shape[2],
                       dtype=init_logits.dtype,
                       device=device))

    if args.optimize_hidden_states:
        y_states = init_states
        epsilon = torch.nn.Parameter(torch.zeros_like(y_states))
    else:
        y_logits = init_logits
        epsilon = torch.nn.Parameter(torch.zeros_like(y_logits))
    if args.prefix_length > 0:
        optim = torch.optim.Adam([epsilon, prefix_logits], lr=args.stepsize)
    else:
        optim = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                step_size=args.stepsize_iters,
                                                gamma=args.stepsize_ratio)

    frozen_len = args.frozen_length

    noise_std = 0.0

    # Encode x (prompt) beforehand
    assert args.prefix_length <= 0, "The current code does not support prefix-length > 0"
    # The last token of x (prompt) is used in soft_forward; [B, 1, V]
    soft_forward_x = x_onehot[:, -1:, :]
    if x_encoded.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_encoded[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        x_model_past = [(p[0].detach(), p[1].detach()) if isinstance(p, tuple) else p.detach()
                        for p in x_model_past]

    mask_t = None  # NOTE: mask_t is the top-k logit mask of the base LM.

    for it in range(args.num_iters):
        optim.zero_grad()
        if args.optimize_hidden_states:
            y_states_t = y_states + epsilon
            y_logits_t = model.lm_head(y_states_t)
        else:
            y_logits_t = y_logits + epsilon

        # TODO: optimize_hidden_states
        mask_t, fluency_loss = fluency_constraint(
            model,
            args,
            y_logits_t,
            soft_forward_x,
            x_model_past,
            mask_t=mask_t,
            model_back=model_back,
            z_mask=z_mask,
            z_onehot=z_onehot,
            temperature=args.fluency_temp,
            straight_through=args.straight_through,
        )
        # fluency_loss = torch.zeros(args.batch_size).to(device)  # for debugging

        # n-gram similarity constraint
        constraint_loss = {}
        if "sentence_ngram" in constraint_functions and args.sentence_ngram_weight > 0:
            filtered_y_logits = top_k_filter_3d(y_logits_t,
                                                args.topk,
                                                mask=mask_t,
                                                extra_mask=z_mask)
            # right-context n-gram similarity constraint
            sent_ngram_loss = sentence_ngram_similarity_constraint(
                filtered_y_logits, z_encoded, max_ngram=args.counterfactual_max_ngram)
            constraint_loss["sentence_ngram"] = sent_ngram_loss * args.sentence_ngram_weight

        if "right_context_pred" in constraint_functions and args.right_context_pred_weight > 0:
            # right-context prediction constraint
            # TODO: optimize_hidden_states
            r_pred_loss = right_context_pred_constraint(
                model,
                args,
                z_encoded,
                z_onehot,
                y_logits_t,
                soft_forward_x,
                temperature=args.constraint_temp,
            )
            constraint_loss["right_context_pred"] = r_pred_loss * args.right_context_pred_weight

        if "keyword" in constraint_functions and args.keyword_weight > 0:
            # keyword similarity (inclusion; 1-gram based bleu) constraint
            kw_loss = keyword_lexical_constraint(y_logits_t, keywords_encoded)
            constraint_loss["keyword"] = kw_loss * args.keyword_weight

        elif "keyword_sg" in constraint_functions and args.keyword_weight > 0:
            # keyword constraint using scale grad loss
            kw_sg_loss = keyword_sg_constraint(
                model,
                args,
                y_logits_t,
                soft_forward_x,
                x_model_past,
                mask_t=mask_t,
                z_mask=z_mask,
                temperature=args.fluency_temp,
                straight_through=args.straight_through,
            )
            constraint_loss["keyword"] = kw_sg_loss * args.keyword_weight

        if "attr_control" in constraint_functions and args.attr_control_weight > 0:
            # attribute control with classifer gradients
            # TODO: optimize_hidden_states
            attr_control_loss = attr_control_constraint(
                model,
                args,
                y_logits_t,
                soft_forward_x,
                x_model_past,
                mask_t=None if args.no_fluency_mask_in_attr_control else mask_t,
                z_mask=z_mask,
                pool_method=args.pool_method,
                attribute_class_idx=args.attr_cls_idx,
                temperature=args.constraint_temp,
                straight_through=args.straight_through,
            )
            constraint_loss["attr_control"] = attr_control_loss * args.attr_control_weight

        if "selfcond" in constraint_functions and args.selfcond_weight > 0 and args.selfcond_mode == 'constraint':
            expert_loss = expert_activation_constraint(
                model_wrapper,
                soft_forward_x,
                y_logits_t,
                x_model_past,
                expert_per_layer,
                args,
                only_last_token=only_last_token,
                mask_t=mask_t,
                z_mask=z_mask,
                temperature=args.constraint_temp,
                straight_through=args.straight_through,
            )

            constraint_loss["selfcond"] = expert_loss * args.selfcond_weight

        c_loss = sum(constraint_loss.values())

        loss = (1.0 - args.constraint_weight) * fluency_loss + args.constraint_weight * c_loss
        loss = loss.mean()

        if it < args.num_iters - 1:  # so that the mask_t at the last iteration will not change
            loss.backward()
            # print(f"[DEBUG]: iter: {it}, gradient at:")
            # print(torch.ne(epsilon.grad.sum(-1), 0).int())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([epsilon], args.grad_clip)
            optim.step()
            scheduler.step()  # turn off the scheduler
            last_lr = scheduler.get_last_lr()[0]

        if (args.verbose or args.wandb) and ((it + 1) % args.print_every == 0 or it == 0 or
                                             it + 1 == args.num_iters):
            if args.discretize_method == 'raw':
                text, ppl, _ = get_text_from_logits(
                    top_k_filter_3d(y_logits_t, args.topk, extra_mask=z_mask),
                    tokenizer,
                    topk=args.topk)
            elif args.discretize_method == 'geometric':
                text, ppl, _ = geometric_mean_fusion(model,
                                                     y_logits_t,
                                                     args.topk,
                                                     soft_forward_x,
                                                     x_model_past,
                                                     tokenizer,
                                                     gamma=args.geometric_gamma,
                                                     extra_mask=z_mask)
            else:
                assert args.discretize_method in ('cold_over_model', 'model_over_cold')
                topk_from_model = args.discretize_method == 'model_over_cold'
                text, ppl, _ = decode_with_model_topk(model,
                                                    y_logits_t,
                                                    args.topk,
                                                    soft_forward_x,
                                                    x_model_past,
                                                    tokenizer,
                                                    extra_mask=z_mask,
                                                    topk_from_model=topk_from_model)
            for bi in range(args.batch_size):
                if args.verbose:
                    print(f"{it + 1}, loss: {loss.item():.4f}, "
                          f"fluency_loss: {fluency_loss[bi].item():.4f}, "
                          f"c_loss: {c_loss[bi].item():.4f}, "
                          f"ppl: {ppl[bi]:.4f}, "
                          f"lr: {last_lr:.4f}, |{text[bi]}|")
                if args.wandb:
                    text_table.add_data(it + 1, prompt, keyword_constraint, text[bi], ppl[bi].item())

        if args.wandb:
            log_items = {
                "Loss": loss.item(),
                'fluency loss': {f'gen-{i}': v.item() for i, v in enumerate(fluency_loss, 1)},
                'constraint loss': {f'gen-{i}': v.item() for i, v in enumerate(c_loss, 1)},
                "Gassian_Noise_STD": noise_std,
                "LR": last_lr,
                "Gradient": torch.norm(epsilon.grad).detach().clone().data.cpu().numpy()
            }
            experiment.log(log_items)

        # noise
        if it < args.num_iters - 1:

            if 'grammar' in args.mode:
                continue

            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            if it % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if it < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean,
                                     std=noise_std,
                                     size=epsilon.size(),
                                     device=device,
                                     requires_grad=False)
                if args.win_anneal_iters >= 0 and it >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    if args.discretize_method == 'raw':
        text, ppl, last_text_ids = get_text_from_logits(
            top_k_filter_3d(y_logits_t, args.topk, extra_mask=z_mask),
            tokenizer,
            topk=args.topk)
    elif args.discretize_method == 'geometric':
        text, ppl, last_text_ids = geometric_mean_fusion(model,
                                             y_logits_t,
                                             args.topk,
                                             soft_forward_x,
                                             x_model_past,
                                             tokenizer,
                                             gamma=args.geometric_gamma,
                                             extra_mask=z_mask)
    else:
        assert args.discretize_method in ('cold_over_model', 'model_over_cold')
        topk_from_model = args.discretize_method == 'model_over_cold'
        text, _, last_text_ids = decode_with_model_topk(model,
                                                        y_logits_t,
                                                        args.topk,
                                                        soft_forward_x,
                                                        x_model_past,
                                                        tokenizer,
                                                        extra_mask=z_mask,
                                                        topk_from_model=topk_from_model)

    last_logits = model(input_ids=last_text_ids).logits
    last_rank_loss = lm_score_from_logits(last_logits,
                                          last_text_ids).detach().clone().data.cpu().numpy()
    text_post = post_process(last_text_ids, model, args.max_length, args.length, tokenizer, device)
    ppl_last = np.exp(last_rank_loss)

    if args.verbose or args.wandb:
        for txt, post, ppl in zip(text, text_post, ppl_last):
            if args.verbose:
                print(f"[final]: {txt}\n{ppl:.4f}")
                print(f"[final complete sentence]: {post}\n")
            if args.wandb:
                text_table.add_data(args.num_iters + 1, prompt, keyword_constraint, post, ppl)

    if args.wandb:
        experiment.log({"generated texts": text_table})
        wandb.finish()

    return ppl_last, text, text_post


def counterfactual_reasoning(model,
                             tokenizer,
                             data,
                             args,
                             model_back=None,
                             device='cpu',
                             outfile='output.json',
                             **expert_kwargs):
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')
    fw_res = open(os.path.join(args.output_dir, 'res_' + outfile), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        if args.seed != -1:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        print(f"{i} / {len(data)}")
        print('output-lgt-temp:\t', args.output_lgt_temp)

        premise = d.get('premise', "")
        counterfactual = d.get('counterfactual', "")

        x = premise + ' ' + counterfactual
        ori_ending = d.get('original_ending', "")
        ori_endings = tokenize.sent_tokenize(ori_ending)

        if x not in procssed:
            procssed.add(x)

        x_text_so_far = [""]
        x_addon = [[x]]

        outputs = []
        for oi, z_sent in enumerate(ori_endings):  # per sentence in original ending
            print(f"Sentence {oi}")
            z_text_so_far = ". " + z_sent.strip()

            assert len(x_text_so_far) == len(x_addon), f"{len(x_text_so_far)} vs {len(x_addon)}"

            new_x_text_so_far = []
            new_x_addon = []
            for ii, text_i in enumerate(x_text_so_far):
                for text_j in x_addon[ii]:
                    text_ij = text_i.strip() + " " + text_j.strip()
                    new_x_text_so_far.append(text_ij)

                    text_ij = text_ij.strip()

                    ppl_last, text, text_post = decode(model,
                                                       tokenizer,
                                                       args,
                                                       prompt=text_ij,
                                                       sent_constraint=z_text_so_far,
                                                       constraint_functions=('sentence_ngram',),
                                                       device=device,
                                                       model_back=model_back,
                                                       **expert_kwargs)

                    outputs.append([text_ij, text_post])

                    #  Rank and filter text_post from util.py:
                    text_post = [post_sent(x) for x in text_post]
                    if not isinstance(model, nn.Module):
                        torch_module = model.module
                    else:
                        torch_module = model
                    text_post = rank_and_filter(text_post, text_ij, z_text_so_far, torch_module,
                                                tokenizer, device, args.no_loss_rerank)

                    if ii == len(x_text_so_far) - 1 and oi == len(ori_endings) - 1:
                        last_output = text_post
                        final_res = ' '.join([text_ij, last_output])
                        outputs.append(final_res)
                        fw_res.write(final_res + '\n')
                        fw_res.flush()

                    new_x_addon.append([text_post])

            x_text_so_far = new_x_text_so_far
            x_addon = new_x_addon

            # break

        complete_output = outputs
        out = {
            'premise': premise,
            'initial': d.get('initial', ""),
            'counterfactual': counterfactual,
            'original_ending': ori_ending,
            'generation_complete': complete_output,
        }

        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print(f"outputs: {outfile}")

def abductive_reasoning_t5(model,
                           tokenizer,
                           data,
                           args,
                           model_back=None,
                           device='cpu',
                           outfile='output.json',
                           **expert_kwargs):
    fw = open(os.path.join(args.output_dir, outfile), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        x = d["obs1"].strip()[:-1]  # remove last punctuationb
        z = d["obs2"].strip()

        if ' '.join([x, z]) in procssed:
            continue
        procssed.add(' '.join([x, z]))

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            encoded = tokenizer(f"{x} <extra_id_0> {z}", return_tensors="pt").to(device)
            # sequence_ids = model.generate(**encoded, do_sample=True, top_k=args.topk, max_new_tokens=args.max_length)
            sequence_ids = model.generate(**encoded, max_new_tokens=args.max_length)

            # sequences
            # ['<pad> <extra_id_0> park offers<extra_id_1> the<extra_id_2> park.</s>']
            # need to parse
            # extra_ids = tokenizer.encode('<extra_id_0><extra_id_1>')
            # first_id = torch.where(sequence_ids == extra_ids[0])[1]
            # second_id = torch.where(sequence_ids == extra_ids[1])[1]
            # roi_ids = sequence_ids[:, first_id + 1:second_id]

            text = tokenizer.batch_decode(sequence_ids)
            # text_post = post_process(roi_ids, model, args.max_length, args.length, tokenizer, device)
            text_post = text
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x': x,
            'z': z,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }

        fw.write(json.dumps(out) + '\n')
        fw.flush()

    print(f"outputs: {outfile}")

def abductive_reasoning(model,
                        tokenizer,
                        data,
                        args,
                        model_back=None,
                        device='cpu',
                        outfile='output.json',
                        **expert_kwargs):
    fw = open(os.path.join(args.output_dir, outfile), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        x = d["obs1"].strip()
        z = d["obs2"].strip()
        z_keywords = get_keywords(z, d["obs1"].strip(), args)

        if ' '.join([x, z]) in procssed:
            continue
        procssed.add(' '.join([x, z]))

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z

        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            ppl_last, text, text_post = decode(model,
                                               tokenizer,
                                               args,
                                               prompt=x,
                                               sent_constraint=z,
                                               keyword_constraint=z_keywords,
                                               constraint_functions=('right_context_pred',
                                                                     'keyword'),
                                               device=device,
                                               model_back=model_back,
                                               **expert_kwargs)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x': x,
            'z': z,
            'z_keywords': z_keywords,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }

        fw.write(json.dumps(out) + '\n')
        fw.flush()

    print(f"outputs: {outfile}")


def lexical_generation(model,
                       tokenizer,
                       data,
                       args,
                       model_back=None,
                       device='cpu',
                       outfile='output.json',
                       **expert_kwargs):
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')
    wandb_runname_base = args.wandb_runname
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print(d["concept_set"])
        constraints = ' '.join(d["concept_set"].split("#"))

        print(f"{i} / {len(data)}")
        text_candidates = []
        text_complete_candidates = []
        for j in range(args.repeat_batch):
            args.wandb_runname = f"{wandb_runname_base}-sent{i}-trial{j}"
            if args.force_tokens:
                force_tokens = tokenizer.encode(constraints)
                sequence_ids = model.generate(max_new_tokens=args.max_length,
                                              force_words_ids=[force_tokens],
                                              num_beams=4)
                text = tokenizer.batch_decode(sequence_ids)
                text_post = post_process(sequence_ids, model, args.max_length, args.length, tokenizer, device)
                text_candidates.extend(text)
                text_complete_candidates.extend(text_post)
            else:
                ppl_last, text, text_post = decode(model,
                                                   tokenizer,
                                                   args,
                                                   prompt=None,
                                                   sent_constraint=". " + constraints,
                                                   keyword_constraint=constraints,
                                                   constraint_functions=('right_context_pred',
                                                                         'keyword_sg'),
                                                   device=device,
                                                   model_back=model_back,
                                                   **expert_kwargs)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'constraints': constraints,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }
        print(out)

        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print(f"outputs: {outfile}")


def prompted_generation(model,
                        tokenizer,
                        prompt,
                        args,
                        device='cpu',
                        outfile='output.json',
                        **expert_kwargs):
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    text_candidates = []
    text_complete_candidates = []
    constraint_fns = ('attr_control',) if args.use_attribute_classifer else tuple()
    for _ in range(args.repeat_batch):
        ppl_last, text, text_post = decode(model,
                                           tokenizer,
                                           args,
                                           prompt=prompt,
                                           sent_constraint=None,
                                           keyword_constraint=None,
                                           constraint_functions=constraint_fns,
                                           device=device,
                                           **expert_kwargs)
        text_candidates.extend(text)
        text_complete_candidates.extend(text_post)

    out = {
        'prompt': prompt,
        'generation': text_candidates,
        'generation_complete': text_complete_candidates,
    }
    print(out)

    fw_pretty.write(json.dumps(out, indent=4) + '\n')
    fw_pretty.flush()

    print(f"outputs: {outfile}")


def main():
    args = options()
    if args.optimize_hidden_states:
        # TODO
        raise NotImplementedError("Not implemented yet!")
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if args.seed != -1:
        set_random_seeds(args.seed)
    # Load pretrained model
    if 't5' in args.pretrained_model:
        model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model)
        model = model.to(device)
        model.eval()
        freeze_module(model)
    elif args.selfcond:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache/huggingface/hub/')
        model = PytorchTransformersModel(args.pretrained_model,
                                         cache_dir,
                                         seq_len=args.length,
                                         device=device)
        model.module.eval()
        # Freeze PTLM weights
        freeze_module(model.module)
    elif args.use_attribute_classifer:
        model = DoubleHeadModel.from_pretrained(args.pretrained_model,
                                                output_hidden_states=True,
                                                resid_pdrop=0,
                                                embd_pdrop=0,
                                                attn_pdrop=0,
                                                summary_first_dropout=0,
                                                num_labels=args.num_attributes)
        model.score.load_state_dict(torch.load(args.attribute_classifier_path))
        model = model.to(device)
        model.eval()
        # Freeze GPT-2 weights
        freeze_module(model)
        args.mode += '-attr_control'
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model,
                                                output_hidden_states=True,
                                                resid_pdrop=0,
                                                embd_pdrop=0,
                                                attn_pdrop=0,
                                                summary_first_dropout=0)
        model = model.to(device)
        model.eval()
        # Freeze GPT-2 weights
        freeze_module(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    if args.use_back_model:
        from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
        model_back = OpenGPT2LMHeadModel.from_pretrained(args.back_model,
                                                         hidden_dropout_prob=0,
                                                         attention_probs_dropout_prob=0,
                                                         summary_first_dropout=0)
        model_back.to(device)
        model_back.eval()
        # Freeze GPT-2 weights
        freeze_module(model_back)
    else:
        model_back = None

    # selfcond
    if args.selfcond:
        args.mode += '-selfcond'
        expertise_df = pd.read_csv(args.expertise)
    else:
        expertise_df = None
    expert_kwargs = {
        'expertise': expertise_df,
        'value': args.forcing,
        'metric': args.metric,
        'num_units': args.num_units,
        'top_n': args.top_n,
        'use_layers': (list(expertise_df.sort_values("layer").layer.unique())
                       if expertise_df is not None and args.per_layer else None),
        'only_last_token': args.only_last_token,
    }

    # Load data
    if not os.path.exists(args.input_file):
        print(f'[WARNING]: file {args.input_file} do not exist. Using --prompt.')
        assert args.prompt, "either --input_file or --prompt must be set."
    else:
        with open(args.input_file, 'r', encoding='utf-8') as fr:
            if args.input_file.endswith('.json') or args.input_file.endswith('.jsonl'):
                data = [json.loads(l.strip()) for l in fr.readlines()]
            else:
                raise NotImplementedError("non json files are not supported yet!")

    # output file
    loss_rerank = 'norerank' if args.no_loss_rerank else 'rerank'
    if not args.output_filename:
        args.output_filename = (
            f'{args.version}_seed{args.seed}_{args.start}_{args.end}_{args.mode}'
            f'_cw{args.constraint_weight:.3f}_kwc{args.keyword_weight:.3f}'
            f'_{loss_rerank}_ngram{args.counterfactual_max_ngram}'
            f'_lrnllp{args.lr_nll_portion:.3f}_len{args.length}_topk{args.topk}'
            f'_niter{args.num_iters}_frozlen{args.frozen_length}'
            f'_winiter{args.win_anneal_iters}_noiseiter{args.noise_iters}_gsstd{args.gs_std:.4f}'
            f'_lr{args.stepsize:.3f}_lrratio{args.stepsize_ratio:.2f}'
            f'_lriter{args.stepsize_iters}_{args.large_noise_iters}_{args.large_gs_std}_output.json'
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if "counterfactual" in args.mode:
        exp_run = counterfactual_reasoning
    elif "abductive" in args.mode:
        if "t5" in args.pretrained_model:
            exp_run = abductive_reasoning_t5
        else:
            exp_run = abductive_reasoning
    elif "lexical" in args.mode:
        exp_run = lexical_generation
    else:
        assert args.prompt, "--prompt must be set"
        exp_run = prompted_generation
        data = args.prompt

    exp_run(model,
            tokenizer,
            data,
            args,
            model_back=model_back,
            device=device,
            outfile=args.output_filename,
            **expert_kwargs)


if __name__ == "__main__":
    main()
