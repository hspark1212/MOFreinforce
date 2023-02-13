import torch.nn.functional as F
from torchmetrics.functional import accuracy


def compute_loss(pl_module, batch):
    infer = pl_module.infer(batch)

    tgt_label = infer["tgt_label"]  # [B, seq_len]

    batch_size, _, vocab_dim = infer["output_ol"].shape

    # loss topo
    logit_topo = infer["output_topo"]
    label_topo = tgt_label[:, 0]
    loss_topo = F.cross_entropy(logit_topo, label_topo)
    # loss mc
    logit_mc = infer["output_mc"]
    label_mc = tgt_label[:, 1]
    loss_mc = F.cross_entropy(logit_mc, label_mc)
    # loss ol
    logit_ol = infer["output_ol"].reshape(-1, vocab_dim)
    label_ol = tgt_label[:, 2:].reshape(-1)
    loss_ol = F.cross_entropy(logit_ol, label_ol, ignore_index=0)
    # total loss
    total_loss = loss_topo + loss_mc + loss_ol

    ret = {
        "gen_loss": total_loss,
        "gen_labels": tgt_label,
    }

    # call update() loss and acc
    loss_name = "generator"
    phase = "train" if pl_module.training else "val"
    total_loss = getattr(pl_module, f"{phase}_{loss_name}_loss")(ret["gen_loss"])

    # acc
    acc_topo = getattr(pl_module, f"{phase}_{loss_name}_acc_topo")(
        accuracy(logit_topo.argmax(-1), label_topo)
    )
    acc_mc = getattr(pl_module, f"{phase}_{loss_name}_acc_mc")(
        accuracy(logit_mc.argmax(-1), label_mc)
    )
    acc_ol = getattr(pl_module, f"{phase}_{loss_name}_acc_ol")(
        accuracy(logit_ol.argmax(-1), label_ol, ignore_index=0)
    )

    pl_module.log(f"{loss_name}/{phase}/total_loss", total_loss, batch_size=batch_size, prog_bar=True, sync_dist=True)
    pl_module.log(f"{loss_name}/{phase}/loss_topo", loss_topo, batch_size=batch_size, prog_bar=False, sync_dist=True)
    pl_module.log(f"{loss_name}/{phase}/loss_mc", loss_mc, batch_size=batch_size, prog_bar=False, sync_dist=True)
    pl_module.log(f"{loss_name}/{phase}/loss_ol", loss_ol, batch_size=batch_size, prog_bar=False, sync_dist=True)
    pl_module.log(f"{loss_name}/{phase}/acc_topo", acc_topo, batch_size=batch_size, prog_bar=True, sync_dist=True)
    pl_module.log(f"{loss_name}/{phase}/acc_mc", acc_mc, batch_size=batch_size, prog_bar=True, sync_dist=True)
    pl_module.log(f"{loss_name}/{phase}/acc_ol", acc_ol, batch_size=batch_size, prog_bar=True, sync_dist=True)
    return ret
