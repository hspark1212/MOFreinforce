import math
import torch
import torch.nn as nn


def attention(query, key, value, mask=None, dropout=None):
    """

    :param query: [B, num_heads, seq_len, hid_dim//num_heads]
    :param key: [B, num_heads, seq_len, hid_dim//num_heads]
    :param value: [B, num_heads, seq_len, hid_dim//num_heads]
    :param mask: [B, 1, 1, seq_len]
    :param dropout: (float) dropout_rate
    :return:  [B, num_heads, seq_len, hid_dim//num_heads]
    """
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # hid_dim
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [B, num_heads, seq_len, seq_len]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf'))  # [B, num_heads, seq_len, seq_len]

    p_attn = scores.softmax(dim=-1)  # [B, num_heads, seq_len, seq_len]

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # [B, num_heads, seq_len, hid_dim//num_heads]


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout_rate=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.layer_q = nn.Linear(d_model, d_model)
        self.layer_k = nn.Linear(d_model, d_model)
        self.layer_v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, q, k, v, mask=None):
        # query : [B, seq_len, hid_dim]
        # mask : [B, 1, seq_len] for src, [B, seq_len, seq_len] for tgt
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, seq_len]
        batch_size = q.size(0)

        # [B, seq_len, num_heads, hid_dim//num_heads] -> [B, num_heads, seq_len, hid_dim//num_heads]
        q = self.layer_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.layer_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.layer_k(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, p_attn = attention(
            q, k, v, mask=mask, dropout=self.dropout
        )  # [B, num_heads, seq_len, hid_dim//num_heads]

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)  # [B, seq_len, num_heads, hid_dim//num_heads]
                .contiguous()
                .view(batch_size, -1, self.h * self.d_k)  # [B, seq_len, hid_dim]
        )

        x = self.proj(x)
        return x, p_attn
