import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchmetrics.functional import r2_score


class Predictor(LightningModule):
    def __init__(self, char_dim, mc_dim, topo_dim, embed_dim, hidden_dim):
        super().__init__()
        # parameter
        self.char_dim = char_dim
        self.mc_dim = mc_dim
        self.topo_dim = topo_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # model
        # mc
        self.embedding_mc = nn.Embedding(self.mc_dim, self.embed_dim)
        self.rc_mc = nn.Linear(self.embed_dim, 1)
        # topo
        self.embedding_topo = nn.Embedding(self.topo_dim, self.embed_dim)
        self.rc_topo = nn.Linear(self.embed_dim, 1)
        # ol
        self.embedding_ol = nn.Embedding(self.char_dim, self.embed_dim)
        self.rnn = nn.RNN(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1)
        self.rc_ol = nn.Linear(self.hidden_dim, 1)
        # total
        self.rc_total = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        mc = batch["mc"]
        topo = batch["topo"]
        ol_pad = batch["ol_pad"]
        ol_len = batch["ol_len"]
        # mc
        logit_mc = self.embedding_mc(mc)  # [B, embed_dim]
        logit_mc = self.rc_mc(logit_mc)  # [B, 1]
        # topo
        logit_topo = self.embedding_topo(topo)  # [B, embed_dim]
        logit_topo = self.rc_topo(logit_topo)  # [B, 1]
        # ol
        logit_ol = self.embedding_ol(ol_pad)  # [B, pad_len, embed_dim]
        packed_ol = pack_padded_sequence(logit_ol, ol_len, batch_first=True, enforce_sorted=False)
        output_packed, hidden = self.rnn(packed_ol)  # [B, pad_len, hidden_dim]
        output_ol, len_ol = pad_packed_sequence(output_packed, batch_first=True)
        logit_ol = self.rc_ol(output_ol[:, -1, :])  # [B, 1]

        # total
        logit_total = torch.concat([logit_topo, logit_mc, logit_ol], axis=-1)
        logit_total = self.rc_total(logit_total)
        logit_total = self.sigmoid(logit_total)

        return logit_total

    def training_step(self, batch, batch_idx):
        pred = self(batch).squeeze(-1)
        target = batch["target"]

        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch).squeeze(-1)
        target = batch["target"]

        loss = F.mse_loss(pred, target)
        self.log("val_loss", loss, batch_size=len(target))


    def test_step(self, batch, batch_idx):
        output = {}
        pred = self(batch).squeeze(-1)
        target = batch["target"]

        loss = F.mse_loss(pred, target)
        self.log("test_loss", loss, batch_size=len(target))
        output["pred"] = pred
        output["target"] = target
        return output

    def test_epoch_end(self, outputs):
        # calculate r2
        preds = []
        targets = []
        for output in outputs:
            preds += output["pred"].tolist()
            targets += output["target"].tolist()

        r2 = r2_score(torch.FloatTensor(preds), torch.FloatTensor(targets))
        self.log("test/r2_score", r2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
