import torch.nn.functional as F
from torchmetrics.functional import r2_score


def compute_regression(pl_module, batch, normalizer):
    infer = pl_module.infer(batch)
    batch_size = pl_module.hparams.config["batch_size"]

    logits = pl_module.regression_head(infer["cls_feats"]).squeeze(-1)  # [[B]]
    target = batch["target"]  # [B]

    # normalize encode if config["mean"] and config["std], else pass
    target = normalizer.encode(target)

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


def compute_classification(pl_module, batch):
    infer = pl_module.infer(batch)
    batch_size = pl_module.hparams.config["batch_size"]

    logits, binary = pl_module.classification_head(infer["cls_feats"])  # [B, n_classes]
    target = batch["target"]  # [B]

    assert len(target.shape) == 1

    if pl_module.hparams.config["threshold_classification"] is not None:
        threshold = pl_module.hparams.config["threshold_classification"]
        mask = target < threshold
        target[mask] = 1.
        target[~mask] = 0.
    else:
        pass

    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=target)
    else:
        loss = F.cross_entropy(logits, target.long())

    ret = {
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": target,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classification_loss")(ret["classification_loss"])
    acc = getattr(pl_module, f"{phase}_classification_accuracy")(
        ret["classification_logits"], ret["classification_labels"]
    )

    pl_module.log(f"classification/{phase}/loss", loss, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"classification/{phase}/accuracy", acc, batch_size=batch_size, prog_bar=True)
    return ret
