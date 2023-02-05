from argparse import Namespace
from typing import Union, List, Tuple, Callable, Dict

import torch
from torch import nn

from schedulers import (StepWiseNoiseScheduler, GeometricNoiseScheduler,
                        get_constant_schedule_with_warmup)
from util import in_notebook

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def embedding_quantization(embedding_module: nn.Embedding,
                           inputs_embeds: torch.Tensor,
                           eps: nn.Parameter = None,
                           return_logits: bool = False):
    """
    embedding_module: weight.shape = [V, E]
    inputs_embeds: [B, T, E]
    eps: [B, T, E]
    """
    if eps is not None:
        with torch.no_grad():
            optimized_embeds = inputs_embeds + eps
    else:
        optimized_embeds = inputs_embeds
    # [1, B * T, V]
    distances = torch.cdist(optimized_embeds.view(1, -1, optimized_embeds.shape[-1]),
                            embedding_module.weight.unsqueeze(0))
    # [B, T, V]
    distances = distances.view(optimized_embeds.shape[0], optimized_embeds.shape[1], -1)
    quantized_ids = distances.min(-1).indices  # [B, T]
    quantized_embeds = embedding_module(quantized_ids)  # [B, T, E]

    if eps is not None:
        with torch.no_grad():
            quantized_embeds = quantized_embeds - eps

    if return_logits:
        return quantized_ids, quantized_embeds, -distances

    return quantized_ids, quantized_embeds


def langevin_optimize(model: nn.Module,
                      input_sequence: torch.Tensor,
                      args: Namespace,
                      energy_functions: Dict[str, List[any]],
                      energy_weights: Union[List[Callable], Tuple[Callable]],
                      device: Union[str, torch.device] = 'cpu'):
    model.eval()

    # device management
    input_sequence = input_sequence.to(device)
    eps = nn.Parameter(torch.zeros_like(input_sequence))

    optim = torch.optim.AdamW([eps], lr=args.stepsize)
    scheduler = get_constant_schedule_with_warmup(optim,
                                                  init_lr=args.min_stepsize,
                                                  max_lr=args.stepsize,
                                                  num_warmup_steps=args.warmup_steps,
                                                  num_training_steps=args.num_iters)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
    #                                             step_size=args.stepsize_iters,
    #                                             gamma=args.stepsize_ratio)

    if args.noise_scheduler == 'step-wise':
        noise_schedule = StepWiseNoiseScheduler(input_sequence.size(),
                                                args.large_noise_iters,
                                                args.large_gs_std,
                                                final_std=args.gs_std)
    elif args.noise_scheduler == 'geometric':
        noise_schedule = GeometricNoiseScheduler(input_sequence.size(),
                                                 start=5,
                                                 end=0.05,
                                                 total_steps=args.num_iters,
                                                 anneal_noise_step=100)
    pbar = tqdm(range(args.num_iters), total=args.num_iters)
    history = {
        'energy': [],
        'gradient': [],
        'lr': [],
    }
    for it in pbar:
        optim.zero_grad()
        input_sequence_t = input_sequence + eps

        energies = []
        for name, (energy_func, kwargs) in energy_functions.items():
            if name == 'embedding_fluency' and it > 0:
                kwargs['target_ids'] = quantized_ids
            energy = energy_func(model, input_sequence_t, args, **kwargs)  # [B,]
            energies.append(energy)

        energies = torch.stack(energies, dim=-1)  # [B, num_f]
        weights = torch.FloatTensor(energy_weights).to(device)  # [num_f,]
        loss = energies @ weights  # [B,]
        loss = loss.mean()
        history['energy'].append(loss.item())
        pbar.set_postfix({'loss': loss.item()})

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([eps], args.grad_clip)
        optim.step()
        scheduler.step()
        history['gradient'].append(torch.norm(eps.grad).detach().clone().data.cpu().numpy())
        history['lr'].append(scheduler.get_last_lr()[0])

        # noise
        if it % args.noise_iters == 0:
            noise = noise_schedule.step(scheduler.get_last_lr()[0]).to(device)
            input_sequence += noise

        if args.quantize_embeds and it % args.quantize_every == 0:
            quantized_ids, input_sequence = embedding_quantization(model.get_input_embeddings(),
                                                                  input_sequence,
                                                                  eps=eps)

    return input_sequence + eps, history
