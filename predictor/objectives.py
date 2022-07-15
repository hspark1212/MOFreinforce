import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import precision, recall, r2_score


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def compute_trc(pl_module, batch):
    infer = pl_module.infer(batch)
    batch_size = pl_module.hparams.config["batch_size"]

    logits = pl_module.trc_head(infer["cls_feats"]).squeeze(-1)  # [B]
    target = batch["target"]  # [B]

    # threshold = 0.25
    mask = target < 0.25
    target[mask] = 1.
    target[~mask] = 0.

    loss = F.binary_cross_entropy_with_logits(input=logits, target=target)
    ret = {
        "trc_loss": loss,
        "trc_logits": logits,
        "trc_labels": target,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_trc_loss")(ret["trc_loss"])
    acc = getattr(pl_module, f"{phase}_trc_accuracy")(
        ret["trc_logits"], ret["trc_labels"]
    )
    prec = getattr(pl_module, f"{phase}_trc_precision")(
        precision(preds=logits, target=target.long())
    )
    rec = getattr(pl_module, f"{phase}_trc_recall")(
        recall(preds=logits, target=target.long())
    )

    pl_module.log(f"trc/{phase}/loss", loss, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"trc/{phase}/accuracy", acc, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"trc/{phase}/precision", prec, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"trc/{phase}/recall", rec, batch_size=batch_size, prog_bar=True)
    return ret

def compute_vfr(pl_module, batch):
    infer = pl_module.infer(batch)
    batch_size = pl_module.hparams.config["batch_size"]

    logits = pl_module.vfr_head(infer["cls_feats"]).squeeze(-1)  # [B]
    target = batch["target"]  # [B]
    
    loss = F.mse_loss(logits, target)
    ret = {
        "vfr_loss": loss,
        "vfr_logits": logits,
        "vfr_labels": target,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfr_loss")(ret["vfr_loss"])
    mae = getattr(pl_module, f"{phase}_vfr_mae")(
        F.l1_loss(ret["vfr_logits"], ret["vfr_labels"])
    )
    r2 = getattr(pl_module, f"{phase}_vfr_r2")(
        r2_score(logits, target)
    )

    pl_module.log(f"vfr/{phase}/loss", loss, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"vfr/{phase}/mae", mae, batch_size=batch_size, prog_bar=True)
    pl_module.log(f"vfr/{phase}/r2", r2, batch_size=batch_size, prog_bar=True)
    return ret


def compute_regression(pl_module, batch):
    infer = pl_module.infer(batch)
    batch_size = pl_module.hparams.config["batch_size"]

    logits = pl_module.regression_head(infer["cls_feats"]).squeeze(-1)  # [B]
    target = batch["target"]  # [B]

    loss = F.mse_loss(logits, target)
    ret = {
        "regression_loss": loss,
        "regression_logits": logits,
        "regression_labels": target,
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
    target = torch.LongTensor(batch["target"]).to(logits.device)  # [B]
    assert len(target.shape)

    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=target.float())
    else:
        loss = F.cross_entropy(logits, target)

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
