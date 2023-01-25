import torch

from bleuloss import batch_log_bleulosscnn_ae
from scale_grad import sg_loss
from util import embed_inputs, soft_backward, soft_forward, soft_forward_xyz, soft_nll, top_k_filter_3d


def right_context_pred_constraint(model,
                                  args,
                                  z_encoded,
                                  z_onehot,
                                  y_logits_t,
                                  soft_forward_x,
                                  temperature=0.3):
    """
    z_encoded: the right context. [B, T]
    z_onehot: the right context's one-hot encoded version. [B, T, V]
    y_logits_t: current optimized logit. [B, T, V]
    soft_forward_x: The last token of left context in one-hot mode; [B, 1, V]
    temperature: temperature for the softmax over the current logit `y_logits_t`.
        Default=0.3, which does not result a very sharp distribution.
    """
    soft_forward_y_t = (y_logits_t.detach() / temperature - y_logits_t).detach() + y_logits_t
    xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_t, z_onehot)

    # Reshaping
    bz = args.batch_size
    lg = xyz_logits.shape[1]
    st = xy_length - 1
    ed = xyz_logits.shape[1] - 1
    xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
    z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

    c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(z_logits, z_encoded.view(-1))
    c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)
    return c_loss_1


def fluency_constraint(model,
                       args,
                       y_logits_t,
                       soft_forward_x,
                       x_model_past,
                       mask_t=None,
                       model_back=None,
                       z_mask=None,
                       z_onehot=None,
                       temperature=0.001,
                       straight_through=True):
    """
    y_logits_t: current optimized logit. [B, T, V]
    soft_forward_x: The last token of left context in one-hot mode; [B, 1, V]
    x_model_past: the past kv pairs of the LM cached from the left prompt.
    mask_t: mask for filtering topk logits.
    z_mask: the lexical constraint sequence (or keywords).  [B, T, V]
    z_onehot: the lexical constraint sequence's one-hot encoded version. [B, T, V]
    temperature: temperature for the softmax over the current logit `y_logits_t`.
        Default=0.001, which makes the logit commit only to top 1~2 tokens, by
        making the softmax distribution very sharp.
    """
    soft_forward_y = y_logits_t / temperature
    if straight_through:  # TODO: what does this mean?
        if mask_t is None:
            soft_forward_y = (y_logits_t.detach() / temperature - y_logits_t).detach() + y_logits_t
        else:
            soft_forward_y = top_k_filter_3d(y_logits_t, args.topk, mask=mask_t,
                                             extra_mask=z_mask) / temperature

    y_logits_n = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

    if args.topk == 0:
        mask_t = None
    else:
        _, indices_t = torch.topk(y_logits_n, args.topk)
        mask_t = torch.zeros_like(y_logits_n).scatter_(2, indices_t, 1)

        # Compute loss, gradients, and update.
    lr_nll_loss = soft_nll(
        top_k_filter_3d(y_logits_n / args.output_lgt_temp, args.topk, extra_mask=z_mask),
        y_logits_t / args.input_lgt_temp)

    if args.lr_nll_portion == 1.0 or model_back is None:
        return mask_t, lr_nll_loss

    # add right-to-left model (rl)
    rl_reverse_index = torch.arange(y_logits_t.shape[1] - 1, -1, -1)
    if "counterfactual" in args.mode:
        y_logits_rev = y_logits_t[:, rl_reverse_index, :]
        y_logits_rev_t = model_back(y_logits_rev.argmax(-1) + 1).logits[:, :-1, :]
        y_logits_rev_t = y_logits_rev_t[:, :, 1:y_logits_t.shape[-1] + 1]
        rl_nll_loss = soft_nll(top_k_filter_3d(y_logits_rev_t / args.output_lgt_temp, args.rl_topk),
                               y_logits_rev[:, 1:] / args.input_lgt_temp)
    elif "abductive" in args.mode or "lexical" in args.mode:
        assert z_onehot is not None
        yz_logits_rev = torch.flip(torch.cat([y_logits_t, z_onehot], dim=1), [1])
        yz_logits_rev_t = soft_backward(model_back, yz_logits_rev / 0.00001)
        yz_logits_rev_rev_t = torch.flip(yz_logits_rev_t, [1])
        yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_t.shape[-1] + 1]
        yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_t.shape[1], :]

        tmp_logits = yz_logits_rev_rev_t_
        repetition_mask = torch.cat(
            [torch.softmax(tmp_logits[:, 1:, :], dim=-1),
             torch.zeros_like(tmp_logits[:, -1:, :])],
            dim=1)
        yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4
        yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_.detach()

        rl_nll_loss = soft_nll(
            top_k_filter_3d(yz_logits_rev_rev_t_ / args.rl_output_lgt_temp, args.rl_topk),
            y_logits_t / args.input_lgt_temp)

        fluency_loss = lr_nll_loss + args.lr_nll_portion * rl_nll_loss
    return mask_t, fluency_loss


def keyword_lexical_constraint(y_logits_t, keywords_id):
    """
    y_logits_t: current y_hat logits during Langevin Dynamics.  [B, T, V]
    keywords_id: id of keywords.  [B, K]

    """
    return batch_log_bleulosscnn_ae(decoder_outputs=y_logits_t.transpose(0, 1),
                                    target_idx=keywords_id,
                                    ngram_list=[1])


def keyword_sg_constraint(model,
                          args,
                          y_logits_t,
                          soft_forward_x,
                          x_model_past,
                          mask_t=None,
                          z_mask=None,
                          temperature=0.001,
                          straight_through=True,
                          include_topk_mask=False):
    """
    y_logits_t: current optimized logit. [B, T, V]
    soft_forward_x: The last token of left context in one-hot mode; [B, 1, V]
    x_model_past: the past kv pairs of the LM cached from the left prompt.
    mask_t: mask for filtering topk logits.
    z_mask: the lexical constraint sequence (or keywords).  [B, T, V]
    temperature: temperature for the softmax over the current logit `y_logits_t`.
        Default=0.001, which makes the logit commit only to top 1~2 tokens, by
        making the softmax distribution very sharp.
    """
    soft_forward_y = y_logits_t / temperature
    if straight_through:  # TODO: what does this mean?
        if mask_t is None:
            soft_forward_y = (y_logits_t.detach() / temperature - y_logits_t).detach() + y_logits_t
        else:
            soft_forward_y = top_k_filter_3d(y_logits_t, args.topk, mask=mask_t,
                                             extra_mask=z_mask) / temperature

    y_logits_n = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

    if args.topk == 0:
        mask_t = None
    else:
        _, indices_t = torch.topk(y_logits_n, args.topk)
        mask_t = torch.zeros_like(y_logits_n).scatter_(2, indices_t, 1)

    # Compute loss, gradients, and update.
    filtered_model_logits = top_k_filter_3d(y_logits_n / args.output_lgt_temp,
                                            args.topk,
                                            mask=mask_t,
                                            extra_mask=z_mask)

    keywords_mask = z_mask.bool() | mask_t.bool() if include_topk_mask else z_mask.bool()
    loss = sg_loss(
        filtered_model_logits,
        y_logits_t / args.input_lgt_temp,
        keywords_mask,
        args.sg_gamma,
    )

    return loss


def sentence_ngram_similarity_constraint(y_logits_t, target_sent_id, max_ngram=4):
    """
    y_logits_t: current y_hat logits during Langevin Dynamics.  [B, T, V]
    target_sent_id: id of target sentence.  [B, T]
    max_ngram: max number of ngram size.
    """
    ngram_sim = batch_log_bleulosscnn_ae(decoder_outputs=y_logits_t.transpose(0, 1),
                                         target_idx=target_sent_id,
                                         ngram_list=list(range(2, max_ngram + 1)))
    return ngram_sim


def expert_activation_constraint(model_wrapper,
                                 soft_forward_x,
                                 y_logits_,
                                 x_model_past,
                                 expert_per_layer,
                                 args,
                                 only_last_token=False,
                                 mask_t=None,
                                 z_mask=None,
                                 temperature=0.3,
                                 straight_through=True):
    """
    temperature: temperature for the softmax over the current logit `y_logits_t`.
        Default=0.3, which does not result a very sharp distribution.
    """
    # get response of all expert neurons
    soft_forward_y = y_logits_ / temperature
    if straight_through:
        if mask_t is None:
            soft_forward_y = (y_logits_.detach() / temperature - y_logits_).detach() + y_logits_
        else:
            soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t,
                                             extra_mask=z_mask) / temperature

    xy_embeds = embed_inputs(model_wrapper.module.get_input_embeddings().weight,
                             soft_forward_y,
                             x_onehot=soft_forward_x,
                             device=soft_forward_x.device)
    input_batch = {'past_key_values': x_model_past, 'inputs_embeds': xy_embeds}
    # collect response during soft forward
    response_batch = model_wrapper.run_inference(inputs=input_batch,
                                                 outputs=expert_per_layer.keys(),
                                                 trainable=True)

    # all_vals.shape = [n_layer, U]
    references = []
    current_activations = []
    for layer_name, val in expert_per_layer.items():
        unit = val['units']
        ref = val['values']
        if only_last_token:
            # [B, U]
            curr_expert_act = response_batch[layer_name][:, -1, unit]
            ref = ref.view(1, -1)  # [1, U]
        else:
            # [B, T, U]
            curr_expert_act = response_batch[layer_name][:, :, unit]
            ref = ref.view(1, 1, -1)  # [1, 1, U]
        current_activations.append(curr_expert_act)
        references.append(ref)

    current_activations = torch.cat(current_activations, -1)  # [B, U] or [B, T, U]
    references = torch.cat(references, -1)  # [1, U] or [1, 1, U]
    references = references.to(current_activations.device)

    expert_loss = torch.nn.functional.mse_loss(current_activations, references, reduction='none')
    expert_loss = expert_loss.mean(-1)
    if len(expert_loss.shape) > 1:
        expert_loss = expert_loss.mean(-1)
    return expert_loss


def attr_control_constraint(model,
                            args,
                            y_logits_t,
                            soft_forward_x,
                            x_model_past,
                            mask_t=None,
                            z_mask=None,
                            pool_method='last',
                            attribute_class_idx=1,
                            temperature=0.3,
                            straight_through=True):
    """
    y_logits_t: current optimized logit. [B, T, V]
    soft_forward_x: The last token of left context in one-hot mode; [B, 1, V]
    x_model_past: the past kv pairs of the LM cached from the left prompt.
    mask_t: mask for filtering topk logits.
    z_mask: the lexical constraint sequence (or keywords).  [B, T, V]
    z_onehot: the lexical constraint sequence's one-hot encoded version. [B, T, V]
    temperature: temperature for the softmax over the current logit `y_logits_t`.
        Default=0.3, which does not result a very sharp distribution.
    """
    soft_forward_y = y_logits_t / temperature
    if straight_through:
        if mask_t is None:
            soft_forward_y = (y_logits_t.detach() / temperature - y_logits_t).detach() + y_logits_t
        else:
            soft_forward_y = top_k_filter_3d(y_logits_t, args.topk, mask=mask_t,
                                             extra_mask=z_mask) / temperature

    y_logits_n, classifier_logits = soft_forward(model,
                                                 soft_forward_x,
                                                 soft_forward_y,
                                                 x_past=x_model_past,
                                                 return_scorer=True,
                                                 pool_method=pool_method)

    classifier_probs = torch.softmax(classifier_logits, dim=-1)
    # since we want to maximize this, negative here
    return 1 - classifier_probs[:, attribute_class_idx]


# embedding space
def embedding_fluency_constraint(model,
                                 inputs_embeds,
                                 args,
                                 left_context_ids=None,
                                 right_context_ids=None,
                                 is_encoder_decoder=False,
                                 temperature=1.0):
    del args  # unused
    # extra input context management
    if left_context_ids is not None:
        if is_encoder_decoder:
            encoder_input_ids = left_context_ids
        else:
            left_embed = model.get_input_embeddings()(left_context_ids).detach()
            inputs_embeds = torch.cat([left_embed, inputs_embeds], dim=1)
    if right_context_ids is not None:
        right_embed = model.get_input_embeddings()(right_context_ids).detach()  # [B, T, E]
        inputs_embeds = torch.cat([inputs_embeds, right_embed], dim=1)

    # model forward
    if is_encoder_decoder:
        outs = model(input_ids=encoder_input_ids,
                     decoder_inputs_embeds=inputs_embeds,
                     output_hidden_states=True)
        model_logits = outs.logits  # [B, T, V]
        hidden_states = outs.decoder_hidden_states[-1]  # [B, T, E]
        if model.config.tie_word_embeddings:
            hidden_states = hidden_states * (model.model_dim**-0.5)
    else:
        outs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        model_logits = outs.logits  # [B, T, V]
        hidden_states = outs.hidden_states[-1]  # [B, T, E]


    # NOTE: dotplusplus from mucola
    opt_start_idx = 1
    opt_end_idx = 0
    if not is_encoder_decoder and left_context_ids is not None:
        opt_start_idx = left_context_ids.shape[1]
    # if right_context_ids is not None:
    #     opt_end_idx = -right_context_ids.shape[1]
    opt_hidden_states = hidden_states[:, opt_start_idx - 1:opt_end_idx - 1]  # [B, T - 1, E]
    opt_model_logits = model_logits[:, opt_start_idx - 1:opt_end_idx - 1]    # [B, T - 1, V]
    opt_embeds = inputs_embeds[:, opt_start_idx:opt_end_idx or None]   # [B, T - 1, V]

    # compute loss
    if model.config.tie_word_embeddings:
        logprob = (opt_hidden_states * opt_embeds).sum(dim=-1) / temperature

        maxlogit, maxlogit_idx = opt_model_logits.max(dim=-1, keepdim=False)
        logits_sub_max = opt_model_logits - maxlogit.unsqueeze(-1)

        # ver1 (original)  # NOTE: exp() are unstable
        # grad_version = (opt_hidden_states * opt_embeds).sum(dim=-1) / temperature - maxlogit
        # detached = (opt_hidden_states * opt_embeds.detach()).sum(dim=-1) / temperature - maxlogit
        # additive = grad_version.exp() - detached.exp()
        # lognorm = torch.log(logits_sub_max.exp().sum(dim=-1) + additive) + maxlogit

        # ver2 (stable)
        # lognorm = torch.logsumexp(opt_model_logits, dim=-1)

        # ver3 (unstable)
        # log_p = (opt_hidden_states * opt_embeds).sum(dim=-1) / temperature
        # lognorm = torch.log(opt_model_logits.exp().sum(-1) + log_p.exp() - maxlogit.exp())

        # ver4 (stable)
        # log_p = (opt_hidden_states * opt_embeds).sum(dim=-1) / temperature - maxlogit
        # lognorm = torch.logsumexp(
        #     torch.cat([logits_sub_max, log_p.unsqueeze(-1)], dim=-1), dim=-1) + maxlogit

        # ver5 = substitute (stable)
        log_p = (opt_hidden_states * opt_embeds).sum(dim=-1) / temperature
        bs, t = maxlogit_idx.size()
        idx = (torch.arange(0, bs).repeat_interleave(t),
            torch.arange(0, t).repeat(bs),
            maxlogit_idx.view(-1))
        opt_model_logits[idx] = log_p
        lognorm = torch.logsumexp(opt_model_logits, dim=-1)


        loss = -logprob + lognorm
        loss = loss.mean(-1)  # [B,]

        # extra NLL Loss
        # if right_context_ids is not None:
        #     right_pred_loss = torch.nn.functional.cross_entropy(
        #         model_logits[:, -right_context_ids.shape[1] - 1:-1].permute(0, 2, 1),
        #         right_context_ids,
        #         reduction='mean')
        #     loss += right_pred_loss
    else:
        raise NotImplementedError

    return loss
