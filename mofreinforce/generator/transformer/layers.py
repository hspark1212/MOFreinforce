import torch
from torch import nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        """
        :param query: [B, seq_len, hid_dim]
        :param key: [B, seq_len, hid_dim]
        :param value: [B, seq_len, hid_dim]
        :param mask: [B, 1, 1, src_len] for src_mask, [B, 1, trg_len, trg_len] for trg_mask
        :return: [B, seq_len, hid_dim], [B, n_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]

        Q = self.fc_q(query) # [batch size, query len, hid dim]
        K = self.fc_k(key) # [batch size, key len, hid dim]
        V = self.fc_v(value) # [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch size, n heads, query len, head dim]
        # [batch size, n heads, key len, head dim]
        # [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        # energy = [batch size, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1) # [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V) # [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous() # [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim) # [batch size, query len, hid dim]
        x = self.fc_o(x) # [batch size, query len, hid dim]
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [B, seq_len, hid dim]
        :return: [B, seq_len, hid_dim]
        """
        x = self.dropout(torch.relu(self.fc_1(x))) # [batch size, seq len, pf dim]
        x = self.fc_2(x) # [batch size, seq len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        :param src: [B, src_len, hid_dim]
        :param src_mask: [B, 1, 1, src_len]
        :return: [B, src_len, hid_dim]
        """
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask) # [batch_size, src_len, hid_dim]

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src)) # [batch_size, src_len, hid_dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src) # [batch_size, src_len, hid_dim]

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src)) # [batch_size, src_len, hid_dim]
        return src


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        :param trg: [B, trg_len, hid_dim]
        :param enc_src: [B, src_len, hid_dim]
        :param trg_mask: [B, 1, trg_len, trg_len]
        :param src_mask: [B, 1, 1, seq_len]
        :return: [B, trg_len, hid_dim], [B, n_heads, trg_len, src_len]
        """

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask) # [batch size, trg len, hid dim]

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg)) # [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # [batch size, trg len, hid dim], [batch size, n heads, trg len, src len]

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg)) # [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg) # [batch size, trg len, hid dim]

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg)) # [batch size, trg len, hid dim]

        return trg, attention