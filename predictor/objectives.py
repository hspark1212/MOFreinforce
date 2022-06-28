import torch.nn.functional as F
from torchmetrics.functional import r2_score, accuracy, precision, recall


def compute_regression(pl_module, pred, target):
    phase = "train" if pl_module.training else "val"
    result = {}

    # calculate loss and metrics
    loss = F.mse_loss(pred, target)
    r2 = r2_score(pred, target)

    # save result
    result[f"{phase}_loss"] = loss
    result[f"{phase}_r2"] = r2
    return result

def compute_classification(pl_module, pred, target):
    phase = "train" if pl_module.training else "val"
    result = {}

    # make binary classification
    mask = target < pl_module.threshold
    target[mask] = 1.
    target[~mask] = 0.

    # calculate loss and metrics
    loss = F.binary_cross_entropy_with_logits(pred, target)
    acc = accuracy(pred, target.long())
    prec = precision(pred, target.long())
    rec = recall(pred, target.long())

    # save result
    result[f"{phase}/loss"] = loss
    result[f"{phase}/acc"] = acc
    result[f"{phase}/precision"] = prec
    result[f"{phase}/recall"] = rec

    return result
