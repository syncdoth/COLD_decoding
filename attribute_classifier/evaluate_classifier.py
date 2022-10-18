import torch


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args, device='cpu'):
    """ Evaluates a given model and dataset.

    obtained from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/evaluate.py
    """
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0

    for inputs, labels in dataloader:
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        yhat = model(inputs).logits
        loss = loss_fn(yhat, labels)

        sample_count += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)  # smaller batches count less
        running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects

    loss = running_loss / sample_count
    acc = running_acc / sample_count

    return loss, acc