import torch.nn.functional as F
from torchmetrics.functional import r2_score


def get_result(pl_module, pred, target):
    phase = "train" if pl_module.training else "val"
    result = {}
    if pl_module.loss_name == "regression":
        r2 = r2_score(pred, target)
        result[f"{phase}_r2"] = r2
    loss = F.mse_loss(pred, target)
    result[f"{phase}_loss"] = loss

    return result
