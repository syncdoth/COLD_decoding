import torch

from util import soft_backward, soft_forward, soft_forward_xyz, soft_nll, top_k_filter_3d


def right_context_pred_constraint(model, args, z_t, z_onehot, y_logits_, soft_forward_x):
    soft_forward_y_ = (y_logits_.detach() / 0.3 - y_logits_).detach() + y_logits_
    xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)

    # Reshaping
    bz = args.batch_size
    lg = xyz_logits.shape[1]
    st = xy_length - 1
    ed = xyz_logits.shape[1] - 1
    xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
    z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

    c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(z_logits, z_t.view(-1))
    c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)
    return c_loss_1


def fluency_constraint(model,
                       args,
                       z_mask,
                       z_onehot,
                       y_logits_,
                       soft_forward_x,
                       x_model_past,
                       rl_reverse_index,
                       mask_t=None,
                       model_back=None):
    soft_forward_y = y_logits_ / 0.001
    if args.straight_through:
        if mask_t is None:
            soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        else:
            soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t,
                                             extra_mask=z_mask) / 0.001

    y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

    if args.topk == 0:
        mask_t = None
    else:
        _, indices_t = torch.topk(y_logits_t, args.topk)
        mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)

        # Compute loss, gradients, and update.
    lr_nll_loss = soft_nll(
        top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=z_mask),
        y_logits_ / args.input_lgt_temp)

    if args.lr_nll_portion == 1.0 or model_back is None:
        rl_nll_loss = lr_nll_loss
    else:
        # add right-to-left model (rl)
        if "counterfactual" in args.mode:
            y_logits_rev = y_logits_[:, rl_reverse_index, :]
            y_logits_rev_t = model_back(y_logits_rev.argmax(-1) + 1).logits[:, :-1, :]
            y_logits_rev_t = y_logits_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
            rl_nll_loss = soft_nll(
                top_k_filter_3d(y_logits_rev_t / args.output_lgt_temp, args.rl_topk),
                y_logits_rev[:, 1:] / args.input_lgt_temp)
        elif "abductive" in args.mode or "lexical" in args.mode:
            yz_logits_rev = torch.flip(torch.cat([y_logits_, z_onehot], dim=1), [1])
            yz_logits_rev_t = soft_backward(model_back, yz_logits_rev / 0.00001)
            yz_logits_rev_rev_t = torch.flip(yz_logits_rev_t, [1])
            yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
            yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]

            tmp_logits = yz_logits_rev_rev_t_
            repetition_mask = torch.cat([
                torch.softmax(tmp_logits[:, 1:, :], dim=-1),
                torch.zeros_like(tmp_logits[:, -1:, :])
            ],
                                        dim=1)
            yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4
            yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_.detach()

            rl_nll_loss = soft_nll(
                top_k_filter_3d(yz_logits_rev_rev_t_ / args.rl_output_lgt_temp, args.rl_topk),
                y_logits_ / args.input_lgt_temp)

    return mask_t, lr_nll_loss, rl_nll_loss
