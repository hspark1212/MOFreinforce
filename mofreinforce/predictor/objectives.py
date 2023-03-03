import torch
import torch.nn.functional as F
from torchmetrics.functional import r2_score

def weighted_mse_loss(logits, target, weight):
    return (weight * (logits - target) ** 2).mean()


def compute_regression(pl_module, batch, normalizer):
    infer = pl_module.infer(batch)
    batch_size = pl_module.hparams.config["batch_size"]

    logits = pl_module.regression_head(infer["cls_feats"]).squeeze(-1)  # [[B]]
    target = batch["target"]  # [B]

    # normalize encode if config["mean"] and config["std], else pass
    target = normalizer.encode(target)

    # weight mse
    weight_loss = pl_module.hparams.config["weight_loss"]
    if weight_loss is not None:
        weight = torch.where(abs(batch["target"]) > abs(weight_loss), 1.0, 1.0 / 100.)
        loss = weighted_mse_loss(logits, target, weight)
    else:
        loss = F.mse_loss(logits, target)

    ret = {
        "regression_loss": loss,
        "regression_logits": normalizer.decode(logits),
        "regression_labels": normalizer.decode(target),
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_regression_loss")(ret["regression_loss"])
    mae = getattr(pl_module, f"{phase}_regression_mae")(
        F.l1_loss(ret["regression_logits"], ret["regression_labels"])
    )

    r2 = getattr(pl_module, f"{phase}_regression_r2")(
        r2_score(logits, target)
    )

    pl_module.log(f"regression/{phase}/loss", loss, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"regression/{phase}/mae", mae, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"regression/{phase}/r2", r2, batch_size=batch_size, prog_bar=True)
    return ret
