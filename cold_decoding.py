#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import time

import nltk
import numpy as np
import torch
import pandas as pd
import wandb

from selfcond.generation import force_units_hooks

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn

from constraints import (attr_control_constraint, expert_activation_constraint, fluency_constraint,
                         keyword_lexical_constraint, right_context_pred_constraint,
                         sentence_ngram_similarity_constraint)
from util import (decode_with_model_topk, freeze_module, get_keywords, get_text_from_logits,
                  initialize, one_hot, post_process, post_sent, rank_and_filter, set_random_seeds,
                  top_k_filter_3d)

from attribute_classifier.attribute_classifier_model import DoubleHeadModel
from selfcond.models import PytorchTransformersModel
from selfcond.expertise import get_expert_reference

stop_words = set(stopwords.words('english'))


def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")  # TODO: meaning?
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    ## experiment
    parser.add_argument("--input-file",
                        type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--output-dir", type=str, default="./data/commongen/")
    parser.add_argument("--use-back-model", action='store_true')
    parser.add_argument("--back-model", type=str, default="danyaljj/opengpt2_pytorch_backward")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--repeat-batch",
                        type=int,
                        default=1,
                        help="loading data util ith examples.")
    parser.add_argument(
        "--mode",
        type=str,
        default='constrained_langevin',
        choices=['lexical_generation', 'counterfactual_langevin', 'abductive_langevin', 'grammar'])
    ## model
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

    assert sent_constraint is not None  # TODO: make this possible
    # delete the "." token we appended before
    z_encoded = torch.tensor(tokenizer.encode(sent_constraint)[1:], dtype=torch.long)
    z_encoded = z_encoded.unsqueeze(0).repeat(args.batch_size, 1)  # [B, T]
    z_onehot = one_hot(z_encoded, dimension=tokenizer.vocab_size)  # [B, T, V]

    length = args.length
    if length <= 0:
        length = z_encoded.shape[1] - length

    # obtain z_mask
    if keyword_constraint is None:
        # NOTE: if no lexical (~keyword based) constraints, obtain keyword from main constraint
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
        keywords_encoded = torch.tensor(tokenizer.encode(keyword_constraint), dtype=torch.long)
        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[keywords_encoded] = 1.
        z_mask = torch.tensor(z_mask)

        # [B, T, V]
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)
        # [B, K]
        keywords_encoded = keywords_encoded.unsqueeze(0).repeat(args.batch_size, 1)

    # device management
    x_encoded = x_encoded.to(device)
    x_onehot = x_onehot.to(device)
    z_encoded = z_encoded.to(device)
    z_onehot = z_onehot.to(device)
    z_mask = z_mask.to(device)
    if keyword_constraint is not None:
        keywords_encoded = keywords_encoded.to(device)

    if args.verbose:
        print(f"prompt:\t|{prompt}|\n"
              f"main constraint:\t|{sent_constraint}|\n"
              f"keyword constraint:\t|{keyword_constraint}|\n"
              f"length:\t{length}")

    # init logits distribution
    if args.init_mode == 'random':
        init_logits = initialize(model, x_encoded, length, args.init_temp, device)
    else:
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
    for text_i in range(text):
        print(f"[initial]: {text_i}")

    if args.wandb:
        wandb.init(project=f'{args.mode}-{round(time.time() * 1000)}', config=args)

    assert args.prefix_length <= 0  # Otherwise not compatible with batch mode

    if args.prefix_length > 0:
        prefix_logits = torch.nn.Parameter(
            torch.rand(x_onehot.shape[0],
                       args.prefix_length,
                       x_onehot.shape[2],
                       dtype=init_logits.dtype,
                       device=device))

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

    ## Encode x beforehand
    assert args.prefix_length <= 0, "The current code does not support prefix-length > 0"
    # The last token of x is used in soft_forward; [B, 1, V]
    soft_forward_x = x_onehot[:, -1:, :]
    if x_encoded.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_encoded[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        x_model_past = [(p[0].detach(), p[1].detach()) if isinstance(p, tuple) else p.detach()
                        for p in x_model_past]

    mask_t = None

    for it in range(args.num_iters):
        optim.zero_grad()
        y_logits_t = y_logits + epsilon

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
        )

        c_loss = 0

        # n-gram similarity constraint
        constraint_loss = {}
        if "sentence_ngram" in constraint_functions and args.sentence_ngram_weight > 0:
            filtered_y_logits = top_k_filter_3d(y_logits_t,
                                                args.topk,
                                                mask=mask_t,
                                                extra_mask=z_mask)
            sent_ngram_loss = sentence_ngram_similarity_constraint(
                filtered_y_logits, z_encoded, max_ngram=args.counterfactual_max_ngram)
            constraint_loss["sentence_ngram"] = sent_ngram_loss * args.sentence_ngram_weight

        if "right_context_pred" in constraint_functions and args.right_context_pred_weight > 0:
            # right-context prediction constraint
            r_pred_loss = right_context_pred_constraint(model, args, z_encoded, z_onehot,
                                                        y_logits_t, soft_forward_x)
            constraint_loss["right_context_pred"] = r_pred_loss * args.right_context_pred_weight

        if "keyword" in constraint_functions and args.keyword_weight > 0:
            # right-context n-gram similarity constraint
            kw_loss = keyword_lexical_constraint(y_logits, keywords_encoded)
            constraint_loss["keyword"] = kw_loss * args.keyword_weight

        if "attr_control" in constraint_functions and args.attr_control_weight > 0:
            # attribute control with classifer gradients
            attr_control_loss = attr_control_constraint(model,
                                                        args,
                                                        y_logits_t,
                                                        soft_forward_x,
                                                        x_model_past,
                                                        mask_t=mask_t,
                                                        z_mask=z_mask,
                                                        pool_method=args.pool_method,
                                                        attribute_class_idx=args.attr_cls_idx)
            constraint_loss["attr_control"] = attr_control_loss * args.attr_control_weight

        if "selfcond" in constraint_functions and args.selfcond_weight > 0 and args.selfcond_mode == 'constraint':
            expert_loss = expert_activation_constraint(model_wrapper,
                                                       soft_forward_x,
                                                       y_logits_t,
                                                       x_model_past,
                                                       expert_per_layer,
                                                       args,
                                                       only_last_token=only_last_token,
                                                       mask_t=mask_t,
                                                       z_mask=z_mask)

            constraint_loss["keyword"] = expert_loss * args.selfcond_weight

        c_loss = sum(constraint_loss.values())

        loss = (1.0 - args.constraint_weight) * fluency_loss + args.constraint_weight * c_loss
        loss = loss.mean()

        if it < args.num_iters - 1:  # so that the mask_t at the last iteration will not change
            loss.backward()
            optim.step()
            scheduler.step()  # turn off the scheduler
            last_lr = scheduler.get_last_lr()[0]

        if args.verbose and ((it + 1) % args.print_every == 0 or it == 0 or
                             it + 1 == args.num_iters):
            text, _, _ = decode_with_model_topk(model,
                                                y_logits_t,
                                                args.topk,
                                                soft_forward_x,
                                                x_model_past,
                                                tokenizer,
                                                extra_mask=z_mask)
            for bi in range(args.batch_size):
                print(f"{it + 1}, loss: {loss.item():.4f}, "
                      f"fluency_loss: {fluency_loss[bi].item():.4f}, "
                      f"c_loss: {c_loss[bi].item():.4f}, "
                      f"lr: {last_lr:.4f}, |{text[bi]}|")

        if args.wandb:
            wandb.log({
                "Loss": loss.item(),
                "fluency loss": fluency_loss.item(),
                "constraint loss": c_loss,
                "Gassian_Noise_STD": noise_std,
                "LR": last_lr,
                "Gradient": torch.norm(epsilon.grad).detach().clone().data.cpu().numpy()
            })

        ## noise
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

    if args.wandb:
        wandb.finish()

    text, _, last_text_ids = decode_with_model_topk(model,
                                                    y_logits_t,
                                                    args.topk,
                                                    soft_forward_x,
                                                    x_model_past,
                                                    tokenizer,
                                                    extra_mask=z_mask)

    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    text_post = post_process(last_text_ids, model, args.max_length, args.length, tokenizer, device)
    ppl_last = np.exp(last_rank_loss)

    if args.verbose:
        for txt, post in zip(text, text_post):
            print(f"[final]: {txt}\n{ppl_last:.4f}")
            print(f"[final complete sentence]: {post}\n")

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
                    text_post = rank_and_filter(text_post, text_ij, z_text_so_far, model, tokenizer,
                                                device, args.no_loss_rerank)

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

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print(d["concept_set"])
        constraints = ' '.join(d["concept_set"].split("#"))

        print(f"{i} / {len(data)}")
        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            ppl_last, text, text_post = decode(model,
                                               tokenizer,
                                               args,
                                               prompt=None,
                                               sent_constraint=". " + constraints,
                                               keyword_constraint=constraints,
                                               constraint_functions=('right_context_pred',
                                                                     'keyword'),
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


def main():
    args = options()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if args.seed != -1:
        set_random_seeds(args.seed)
    # Load pretrained model
    if args.selfcond:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache/huggingface/hub/')
        model = PytorchTransformersModel(args.pretrained_model,
                                         cache_dir,
                                         seq_len=args.length,
                                         device=device)
        model.module.eval()
        # Freeze PTLM weights
        freeze_module(model.module)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model,
                                                output_hidden_states=True,
                                                resid_pdrop=0,
                                                embd_pdrop=0,
                                                attn_pdrop=0,
                                                summary_first_dropout=0)
        model.to(device)
        model.eval()
        # Freeze GPT-2 weights
        freeze_module(model)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

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

    # attribute classifer
    if args.use_attribute_classifer:
        model = DoubleHeadModel(model, num_labels=args.num_attributes)
        state_dict = torch.load(args.attribute_classifier_path)
        score_state_dict = {'score.weight': state_dict['score.weight']}
        model.load_state_dict(score_state_dict, strict=False)
        args.mode += '-attr_control'

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
    with open(args.input_file, 'r', encoding='utf-8') as fr:
        if args.input_file.endswith('.json'):
            data = [json.loads(l.strip()) for l in fr.readlines()]
        else:
            raise NotImplementedError

    # output file
    loss_rerank = 'norerank' if args.no_loss_rerank else 'rerank'
    outfile = (
        f'{args.version}_seed{args.seed}_{args.start}_{args.end}_{args.mode}'
        f'_cw{args.constraint_weight:.3f}_kwc{args.keyword_weight:.3f}'
        f'_{loss_rerank}_ngram{args.counterfactual_max_ngram}'
        f'_lrnllp{args.lr_nll_portion:.3f}_len{args.length}_topk{args.topk}'
        f'_niter{args.num_iters}_frozlen{args.frozen_length}'
        f'_winiter{args.win_anneal_iters}_noiseiter{args.noise_iters}_gsstd{args.gs_std:.4f}'
        f'_lr{args.stepsize:.3f}_lrratio{args.stepsize_ratio:.2f}'
        f'_lriter{args.stepsize_iters}_{args.large_noise_iters}_{args.large_gs_std}_output.json')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if "counterfactual" in args.mode:
        exp_run = counterfactual_reasoning
    elif "abductive" in args.mode:
        exp_run = abductive_reasoning
    elif "lexical" in args.mode:
        exp_run = lexical_generation

    exp_run(model,
            tokenizer,
            data,
            args,
            model_back=model_back,
            device=device,
            outfile=outfile,
            **expert_kwargs)


if __name__ == "__main__":
    main()
