import json
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from rdkit import RDLogger
from tqdm import tqdm

import selfies as sf
RDLogger.DisableLog('rdApp.*')

topo_to_cn = json.load(open("assets/final_topo_cn.json"))
mc_to_cn = json.load(open("assets/mc_cn.json"))

def metrics_generator(pl_module, src, split="val", num_images=20, early_stop=None):
    """
    metrics for generator
    :param pl_module: generator
    :param src: encoded_input
    :param split: val or test
    :param num_images: how many generated organic linkers are tracked in the tensorboard logger.
    :param early_stop: float, early stopping with the accuracy of scaffold
    :return:
    """
    m_valid = 0
    m_conn_match = 0
    list_ol = []
    list_topo_mc = []
    m_scaffold = 0
    list_frags = []
    total_count = len(src)

    for i in tqdm(range(len(src))):
        src_input = src[i].unsqueeze(0)
        with torch.no_grad():
            output = pl_module.evaluate(src_input)

        topo_cn = topo_to_cn[output["topo"]]
        if len(topo_cn) == 1:
            topo_cn.append(2)
        mc_cn = mc_to_cn[output["mc"]]
        ol_cn = output["ol_idx"].count(pl_module.vocab_to_idx["[*]"])
        gen_sm = output["gen_sm"]

        # (1) metric for connection matching
        if set(topo_cn) == set([mc_cn, ol_cn]):
            m_conn_match += 1

        # (2) metric for valid smiles
        # (3) metric for uniqueness ol
        if gen_sm is None:
            continue
        m_valid += 1
        list_ol.append(gen_sm)
        # (4) metric for uniqueness ol
        list_topo_mc.append( "_".join([output["topo"], output["mc"]]))

        # (5) metric for accuracy (BRICS)
        # get frags
        frags = [pl_module.idx_to_vocab[idx.item()] for idx in src_input.squeeze(0)[3:]]
        frags = frags[:frags.index("[EOS]")]
        frags = "".join(frags)

        frags_sm = [sf.decoder(f) for f in frags.split(".")]
        list_frags.append(frags_sm)
        ## replace * with H
        du, hy = Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]')
        try:
            m = Chem.MolFromSmiles(gen_sm)
            check_gen_sm = Chem.ReplaceSubstructs(m, du, hy, replaceAll=True)[0]

            check_ = True
            for sm in frags_sm:
                if not check_gen_sm.HasSubstructMatch(Chem.MolFromSmiles(sm)):
                    check_ = False
            if check_:
                m_scaffold += 1
        except:
            pass

    # add metrics to log
    pl_module.logger.experiment.add_scalar(f"{split}/conn_match", m_conn_match / total_count, pl_module.global_step)
    pl_module.logger.experiment.add_scalar(f"{split}/valid_smiles", m_valid / total_count, pl_module.global_step)
    pl_module.logger.experiment.add_scalar(f"{split}/unique_ol", len(set(list_ol)) / total_count, pl_module.global_step)
    pl_module.logger.experiment.add_scalar(f"{split}/unique_topo_mc", len(set(list_topo_mc)) / total_count, pl_module.global_step)
    pl_module.logger.experiment.add_scalar(f"{split}/scaffold", m_scaffold / total_count, pl_module.global_step)

    assert len(list_ol) == len(list_frags)
    for i in range(num_images):
        idx = random.Random(i).choice(range(len(list_ol)))
        ol = list_ol[idx]
        frags = list_frags[idx]
        imgs = []
        for s in [ol] + frags:
            m = Chem.MolFromSmiles(s)
            if not m:
                continue
            img = MolToImage(m)
            img = np.array(img)
            img = torch.tensor(img)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        pl_module.logger.experiment.add_image(f"{split}/{i}", imgs, pl_module.global_step, dataformats="NHWC")
        if early_stop is not None:
            # callbacks with scaffold
            print(m_scaffold / total_count)
            if m_scaffold / total_count < early_stop:
                raise Exception(f"accuracy of scaffold < {early_stop}")


