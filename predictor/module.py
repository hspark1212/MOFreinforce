import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from predictor import objectives, heads

from predictor.transformer import Transformer
from predictor import module_utils

from torchmetrics.functional import r2_score


class Predictor(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        # build transformer
        self.transformer = Transformer(
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
        )

        # metal node embedding
        self.mc_embedding = nn.Embedding(config["mc_dim"], config["hid_dim"])
        self.mc_embedding.apply(objectives.init_weights)

        # topology embedding
        self.topo_embedding = nn.Embedding(config["topo_dim"], config["hid_dim"])
        self.topo_embedding.apply(objectives.init_weights)

        # organic linker embedding
        self.ol_embedding = nn.Embedding(config["vocab_dim"], config["hid_dim"], padding_idx=0)
        self.ol_embedding.apply(objectives.init_weights)

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # position embedding
        # max_len = ol_max_len (100) + cls + mc + topo
        self.pos_embeddings = nn.Parameter(torch.zeros(1, 100 + 3, config["hid_dim"]))
        # self.pos_embeddings.apply(objectives.init_weights)

        # pooler
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # ===================== loss =====================

        # trc
        if config["loss_names"]["trc"] > 0:
            self.trc_head = heads.TRCHead(config["hid_dim"])
            self.trc_head.apply(objectives.init_weights)

        # vfr
        if config["loss_names"]["vfr"] > 0:
            self.vfr_head = heads.VFRHead(config["hid_dim"])
            self.vfr_head.apply(objectives.init_weights)

        # ===================== Downstream =====================
        hid_dim = config["hid_dim"]

        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        if self.hparams.config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(hid_dim)
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.hparams.config["loss_names"]["classification"] > 0:
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)

        module_utils.set_metrics(self)
        module_utils.set_task(self)
        # self.current_tasks = list()
        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

    def infer(self, batch):
        mc = batch["mc"]  # [B]
        topo = batch["topo"] # [B]
        ol = batch["ol"] # [B, max_len]
        batch_size = len(mc)

        mc_embeds = self.mc_embedding(mc).unsqueeze(1)  # [B, 1, hid_dim]
        topo_embeds = self.topo_embedding(topo).unsqueeze(1)  # [B, 1, hid_dim]

        ol_embeds = self.ol_embedding(ol) # [B, max_len, hid_dim]

        cls_tokens = torch.zeros(batch_size).to(ol_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]

        # total_embedding and mask
        co_embeds = torch.cat(
            [cls_embeds, mc_embeds, topo_embeds, ol_embeds],
            dim=1) # [B, max_len + 3, hid_dim]
        co_masks = torch.cat([torch.ones([batch_size, 3]).to(ol), (ol != 0).float()], dim=1)

        # add pos_embeddings
        final_embeds = co_embeds + self.pos_embeddings
        final_embeds = self.transformer.pos_drop(final_embeds)

        # transformer blocks
        x = final_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)


        x = self.transformer.norm(x)

        cls_feats = self.pooler(x)

        ret = {
            "cls_feats": cls_feats,
            "mc": mc,
            "topo": topo,
            "ol": ol,
            "output": x,
            "output_mask": co_masks,
        }
        return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))

        # trc
        if "trc" in self.current_tasks:
            ret.update(objectives.compute_trc(self, batch))

        # vfr
        if "vfr" in self.current_tasks:
            ret.update(objectives.compute_vfr(self, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = module_utils.Normalizer(self.mean, self.std)
            ret.update(objectives.compute_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.compute_classification(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)
        return output

    def test_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

        # calculate r2 score when regression
        if "regression_logits" in outputs[0].keys():
            logits = []
            labels = []
            for out in outputs:
                logits += out["regression_logits"].tolist()
                labels += out["regression_labels"].tolist()
            r2 = r2_score(torch.FloatTensor(logits), torch.FloatTensor(labels))
            self.log(f"test/r2_score", r2)

    def configure_optimizers(self):
        return module_utils.set_schedule(self)
