import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaselineModel(nn.Module):
    def __init__(self, vocab_dim, mc_dim, topo_dim, embed_dim, hid_dim):
        super(BaselineModel, self).__init__()
        # model
        # mc
        """
        self.embedding_mc = nn.Embedding(self.mc_dim, self.embed_dim)
        self.rc_mc = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
        )
        """
        self.rc_mc = nn.Linear(mc_dim, 1)
        # topo
        self.embedding_topo = nn.Embedding(topo_dim, embed_dim)
        self.rc_topo = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )
        # ol
        self.embedding_ol = nn.Embedding(vocab_dim, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hid_dim, num_layers=1)
        self.rc_ol = nn.Linear(hid_dim, 1)
        # total
        self.rc_total = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        mc = batch["mc"]
        topo = batch["topo"]
        ol_pad = batch["ol_pad"]
        ol_len = batch["ol_len"]
        # mc
        # logit_mc = self.embedding_mc(mc)  # [B, embed_dim]
        # logit_mc = self.rc_mc(logit_mc)  # [B, 1]
        logit_mc = self.rc_mc(mc) # [B, 1]

        # topo
        logit_topo = self.embedding_topo(topo)  # [B, embed_dim]
        logit_topo = self.rc_topo(logit_topo)  # [B, 1]
        # ol
        logit_ol = self.embedding_ol(ol_pad)  # [B, pad_len, embed_dim]
        packed_ol = pack_padded_sequence(logit_ol, ol_len, batch_first=True, enforce_sorted=False)
        output_packed, hidden = self.rnn(packed_ol)  # [B, pad_len, hid_dim]
        output_ol, len_ol = pad_packed_sequence(output_packed, batch_first=True)
        logit_ol = self.rc_ol(output_ol[:, -1, :])  # [B, pad_len, hid_dim] ->[B, 1]

        # total
        logit_total = torch.cat([logit_topo, logit_mc, logit_ol], dim=-1)
        logit_total = self.rc_total(logit_total)
        logit_total = self.sigmoid(logit_total)

        return logit_total
