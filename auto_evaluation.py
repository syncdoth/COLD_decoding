from pprint import pprint

import torch
from torchmetrics.text.bert import BERTScore
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore


def evaluate(preds, target, metrics=('bleu', 'bertscore', 'rougeL'), device='cpu'):
    metric_fn = {}
    if 'bleu' in metrics:
        bleu = BLEUScore()
        metric_fn['bleu'] = bleu
    if 'bertscore' in metrics:
        bertscore = BERTScore(model_name_or_path='bert-base-uncased', device=device, max_length=256, batch_size=16)
        metric_fn['bertscore'] = bertscore
    if 'rougeL' in metrics:
        rouge = ROUGEScore(rouge_keys="rougeL")
        metric_fn['rougeL'] = rouge

    scores = {}
    for metric, func in metric_fn.items():
        scores[metric] = func(preds, target)

    return scores


if __name__ == '__main__':
    preds = ["hello there", "general kenobi"]
    target = [["hello there", "hi there"], ["master kenobi", "general canopi"]]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scores = evaluate(preds, target, device=device)
    pprint(scores)
