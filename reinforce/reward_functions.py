import torch


def get_reward_trc(topo,
                   mc,
                   encoded_sf,
                   predictor,
                   criterion=0,
                   reward_positive=10,
                   reward_negative=3,
                   ):
    pad_encoded_sf = torch.LongTensor([encoded_sf + [0] * (100 - len(encoded_sf))]).cuda()

    mof = {}
    mof["topo"] = topo.unsqueeze(-1)  # [1]
    mof["mc"] = mc.unsqueeze(-1)  # [1]
    mof["ol"] = pad_encoded_sf  # [1, 100]

    output = predictor.infer(mof)
    output = predictor.trc_head(output["cls_feats"])
    output = output.item()
    if output > 0:
        output = 1
    else:
        output = 0.

    if output > criterion:
        reward = reward_positive
    else:
        reward = reward_negative
    return reward, output


def get_reward_vfr(topo,
                   mc,
                   encoded_sf,
                   predictor,
                   criterion=0.8,
                   reward_positive=10,
                   reward_negative=3,
                   ):
    pad_encoded_sf = torch.LongTensor([encoded_sf + [0] * (100 - len(encoded_sf))]).cuda()

    mof = {}
    mof["topo"] = topo.unsqueeze(-1)  # [1]
    mof["mc"] = mc.unsqueeze(-1)  # [1]
    mof["ol"] = pad_encoded_sf  # [1, 100]

    output = predictor.infer(mof)
    output = predictor.vfr_head(output["cls_feats"])
    output = output.item()
    if output > criterion:
        reward = reward_positive
    else:
        reward = reward_negative
    return reward, output