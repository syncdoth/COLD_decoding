import torch

from bleuloss import batch_log_bleulosscnn_ae
from util import soft_backward, soft_forward, soft_forward_xyz, soft_nll, top_k_filter_3d


def right_context_pred_constraint(model, args, z_encoded, z_onehot, y_logits_t, soft_forward_x):
    """
    z_encoded: the right context. [B, T]
    z_onehot: the right context's one-hot encoded version. [B, T, V]
    y_logits_t: current optimized logit. [B, T, V]
    soft_forward_x: The last token of left context in one-hot mode; [B, 1, V]
    """
    soft_forward_y_t = (y_logits_t.detach() / 0.3 - y_logits_t).detach() + y_logits_t
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
                       z_onehot=None):
    """
    y_logits_t: current optimized logit. [B, T, V]
    soft_forward_x: The last token of left context in one-hot mode; [B, 1, V]
    x_model_past: the past kv pairs of the LM cached from the left prompt.
    mask_t: mask for filtering topk logits.
    z_mask: the lexical constraint sequence (or keywords).  [V,]
    z_onehot: the lexical constraint sequence's one-hot encoded version. [B, T, V]
    """
    soft_forward_y = y_logits_t / 0.001
    if args.straight_through:
        if mask_t is None:
            soft_forward_y = (y_logits_t.detach() / 0.001 - y_logits_t).detach() + y_logits_t
        else:
            soft_forward_y = top_k_filter_3d(y_logits_t, args.topk, mask=mask_t,
                                             extra_mask=z_mask) / 0.001

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