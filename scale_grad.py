import torch
import torch.nn.functional as F


def getNovelMask(target, vocab_size):
    b, l = target.size()
    zeros = torch.zeros(b, l, vocab_size).to(target.device)
    ones = torch.ones(b, l, vocab_size).to(target.device)

    target_index = target.unsqueeze(1).expand(b, l, l).transpose(-2, -1).triu().transpose(-2, -1)
    matrix = zeros.scatter_add_(2, target_index, ones)
    matrix[:, :, 0] = 0
    summ_true = torch.tensor(range(1, l + 1)).unsqueeze(0).float().to(target.device)
    summ_now = torch.sum(matrix, dim=-1)
    diff = summ_true - summ_now
    matrix[:, :, 0] = diff
    matrix = torch.cat((torch.zeros(b, 1, vocab_size).to(target.device), matrix[:, :-1, :]), 1)
    novel_mask = matrix < 1.

    return novel_mask


def sg_loss(logits, target, novel_mask, gamma, mean_batch=True):
    # ScaleGrad
    ##########################################################
    probs = F.softmax(logits, dim=-1)  # [B, T]
    # novel_mask = getNovelMask(target[0].unsqueeze(0), logits.size(-1))

    new_probs = (probs * novel_mask) * gamma + (probs * ~novel_mask) + 1e-8
    new_probs = F.normalize(new_probs, p=1, dim=-1)
    lprobs = torch.log(new_probs)  # [B, T]
    ##########################################################

    loss = F.nll_loss(lprobs, target, reduction='none')  # [B, T]
    loss = loss.sum(-1)  # [B,]
    if mean_batch:
        loss = loss.mean(-1)  # scalar

    return loss
