import torch
from torch import nn

from .layers import EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    """
    Trnasformer for MOF Generator
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 topo_dim,
                 mc_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_len,
                 src_pad_idx,
                 trg_pad_idx,
                 ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            mc_dim=mc_dim,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=pf_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = Decoder(
            output_dim=output_dim,
            topo_dim=topo_dim,
            mc_dim=mc_dim,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=pf_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.apply(self.init_weights)

    def init_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def make_src_mask(self, src):
        """
        make padding mask for src
        :param src: [B, src_len]
        :return: [B, 1, 1, src_len]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        make padding and look-ahead mask for trg
        :param trg: [B, trg_len]
        :return: [B, 1, trg_len, trg_len]
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()  # [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask.to(trg_pad_mask.device)  # [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        """
        :param src: [B, src_len]
        :param trg: [B, trg_len]
        :return: [B, trg_len, vocab_dim] [B, n_heads, trg_len, src_len]
        """
        src_mask = self.make_src_mask(src)  # [batch size, 1, 1, src len]
        trg_mask = self.make_trg_mask(trg)  # [batch size, 1, trg len, trg len]

        # encoder
        enc_src = self.encoder(src, src_mask)  # [batch size, src len, hid dim]

        # decoder
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        # [batch size, trg len, output dim], [batch size, n heads, trg len, src len]
        return output


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 mc_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_len,
                 max_conn=10):
        super().__init__()
        self.mc_embedding = nn.Embedding(mc_dim, hid_dim)
        self.num_embedding = nn.Embedding(max_conn, hid_dim) # num_conn of ol
        self.vocab_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  )
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, src, src_mask):
        """
        :param src: [B, src_len]
        :param src_mask: [B, 1, 1, src_len]
        :param num_conn: [B]
        :return: [B, src_len, hid_dim]
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)  # [batch size, src len]

        src = torch.concat(
            [
                self.mc_embedding(src[:, 0].unsqueeze(-1)),
                self.num_embedding(src[:, 1].unsqueeze(-1)),
                self.vocab_embedding(src[:, 2:]),  # [batch size, src_len, hid dim]
            ],
            dim=1
        )

        src = self.dropout((src * self.scale.to(src.device)) + self.pos_embedding(pos))
        # [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)  # [batch size, src len, hid dim]

        return src


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 topo_dim,
                 mc_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_len,
                 ):
        super().__init__()

        self.topo_embedding = nn.Embedding(topo_dim, hid_dim)
        self.mc_embedding = nn.Embedding(mc_dim, hid_dim)
        self.vocab_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  )
                                     for _ in range(n_layers)])

        self.fc_out_topo = nn.Linear(hid_dim, topo_dim)
        self.fc_out_mc = nn.Linear(hid_dim, mc_dim)
        self.fc_out_ol = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        if len(trg) == 1: topo_embedding
        elif len(trg) == 2: mc_embedding
        else: vocab_embedding
        :param trg: [B, trg_len]
        :param enc_src: [B, src_len, hid_dim]
        :param trg_mask: [B, 1, trg_len, trg_len]
        :param src_mask: [B, 1, 1, src_len]
        :return: [B, trg_len, vocab_dim] [B, n_heads, trg_len, src_len]
        """

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(trg.device)  # [batch size, trg len]

        if trg_len == 1: # [SOS]
            tok_emb = self.vocab_embedding(trg[:, 0].unsqueeze(-1))
        elif trg_len == 2: # [SOS, topo]
            tok_emb = torch.concat(
                    [
                        self.vocab_embedding(trg[:, 0].unsqueeze(-1)),
                        self.topo_embedding(trg[:, 1].unsqueeze(-1)),
                    ],
                    dim=1
                )
        elif trg_len == 3: # [SOS, topo, mc]
            tok_emb = torch.concat(
                [
                    self.vocab_embedding(trg[:, 0].unsqueeze(-1)),
                    self.topo_embedding(trg[:, 1].unsqueeze(-1)),
                    self.mc_embedding(trg[:, 2].unsqueeze(-1)),
                ],
                dim=1
            )
        else: # [SOS, topo, mc, ol]
            tok_emb = torch.concat(
                [
                    self.vocab_embedding(trg[:, 0].unsqueeze(-1)),
                    self.topo_embedding(trg[:, 1].unsqueeze(-1)),
                    self.mc_embedding(trg[:, 2].unsqueeze(-1)),
                    self.vocab_embedding(trg[:, 3:])
                ],
                dim=1
            )

        # [batch size, trg len, hid dim]
        trg = self.dropout((tok_emb * self.scale.to(trg.device)) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # [batch size, trg len, hid dim], [batch size, n heads, trg len, src len]

        output = {}
        if trg_len == 1:
            output.update(
                {
                    "output_topo" : self.fc_out_topo(trg[:, 0]),  # [batch size, topo_dim]
                }
            )
        elif trg_len == 2:
            output.update(
                {
                    "output_topo" : self.fc_out_topo(trg[:, 0]),  # [batch size, topo_dim],
                    "output_mc" : self.fc_out_mc(trg[:, 1]),  # [batch size, mc_dim]
                }
            )
        else:
            output.update(
                {
                    "output_topo" : self.fc_out_topo(trg[:, 0]),  # [batch size, topo_dim],
                    "output_mc" : self.fc_out_mc(trg[:, 1]),  # [batch size, mc_dim]
                    "output_ol" : self.fc_out_ol(trg[:, 2:])  # [batch size, trg len, output dim]
                }
            )
        output.update({"attention" : attention})
        return output
