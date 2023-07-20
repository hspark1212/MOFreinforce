import torch
import numpy as np


def get_attention_score(out, batch_idx, skip_cls=True, normalize=True):
    """
    attention rollout  in "Quantifying Attention Flow in Transformers" paper.
    :param out: output of model.infer(batch)
    :param batch_idx: batch index
    :param skip_cls: <bool> If True, class token is ignored.
    :param normalize: <bool> If True, attention score is normalized.
    :return: <np.ndarray> attention score
    """
    attn_weights = torch.stack(
        out["attn_weights"]
    )  # [num_layers, B, num_heads, max_len, max_len]
    att_mat = attn_weights[:, batch_idx]  # [num_layers, num_heads, max_len, max_len]

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)  # [num_layers, max_len, max_len]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att

    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(
        -1
    )  # [num_layers, max_len, max_len]
    aug_att_mat = aug_att_mat.detach().numpy()  # prevent from memory leakage

    # Recursively multiply the weight matrices
    joint_attentions = np.zeros(aug_att_mat.shape)  # [num_layers, max_len, max_len]
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.shape[0]):
        joint_attentions[n] = np.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]  # [max_len, max_len]

    # skip class token
    if skip_cls:
        v_ = v[0][1:]  # skip cls token
    else:
        v_ = v[0]

    # normalize
    if normalize:
        attention_score = v_ / v_.max()
    else:
        attention_score = v_

    return attention_score
