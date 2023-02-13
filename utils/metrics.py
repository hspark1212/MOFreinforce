import json
import torch
from rdkit import Chem
from rdkit import RDLogger

import libs.selfies as sf

RDLogger.DisableLog('rdApp.*')

topo_to_cn = json.load(open("data/final_topo_cn.json"))
mc_to_cn = json.load(open("data/mc_cn.json"))


class Metrics():
    def __init__(self, vocab_to_idx, idx_to_vocab):
        # generator
        self.num_fail = []
        self.conn_match = []
        self.scaffold = []
        # collect
        self.input_frags = []
        self.gen_ol = []
        self.gen_topo = []
        self.gen_mc = []
        # reinforce
        self.rewards = []
        self.preds = []
        # vocab
        self.topo_to_cn = topo_to_cn
        self.mc_to_cn = mc_to_cn
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = idx_to_vocab

    def update(self, output, src, rewards=None, preds=None):
        """
        (1) metric for connection matching
        (2) metric for unique organic linker
        (3) metric for unique topology and metal cluster
        (4) metric for scaffold
        (5) (optional) rewards
        (6) (optional) preds

        :param output:
        :return:
        """
        topo_cn = topo_to_cn.get(output["topo"], [0])  # if topo is [PAD] then topo_cn = [0]
        if len(topo_cn) == 1:
            topo_cn.append(2)
        mc_cn = mc_to_cn.get(output["mc"], -1)  # if mc is [PAD] then mc_cn = -1
        ol_cn = output["ol_idx"].count(self.vocab_to_idx["[*]"])

        # (1) metric for connection matching
        if set(topo_cn) == {mc_cn, ol_cn}:
            self.conn_match.append(1)
        else:
            self.conn_match.append(0)

        # (2) metric for unique organic linker
        # (3) metric for unique topology and metal cluster
        gen_sm = output["gen_sm"]
        self.gen_ol.append(gen_sm)
        self.gen_topo.append(output["topo"])
        self.gen_mc.append(output["mc"])

        # (4) metric for scaffold
        # get frags
        frags = [self.idx_to_vocab[idx.item()] for idx in src.squeeze(0)[3:]]
        frags = frags[:frags.index("[EOS]")]
        frags = "".join(frags)

        frags_sm = [sf.decoder(f) for f in frags.split(".")]
        self.input_frags.append(frags_sm)
        # replace * with H
        du, hy = Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]')
        try:
            m = Chem.MolFromSmiles(gen_sm)
            check_gen_sm = Chem.ReplaceSubstructs(m, du, hy, replaceAll=True)[0]

            check_ = True
            for sm in frags_sm:
                if not check_gen_sm.HasSubstructMatch(Chem.MolFromSmiles(sm)):
                    check_ = False
            if check_:
                self.scaffold.append(1)
            else:
                self.scaffold.append(0)
        except:
            pass

        # (5) metric for reward
        self.rewards.append(rewards)
        self.preds.append(preds)

    @staticmethod
    def get_mean(list_):
        return torch.mean(torch.Tensor(list_))
