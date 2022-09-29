import json
import numpy as np
from pathlib import Path
import pandas as pd
import selfies as sf

import torch
from torch.utils.data import Dataset


class GeneratorDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 path_vocab,
                 path_topo_to_idx,
                 path_mc_to_idx,
                 split,
                 max_len):
        assert split in ["train", "test", "val"]
        # read vocab_to_idx
        self.vocab_to_idx = json.load(open(path_vocab))
        self.topo_to_idx = json.load(open(path_topo_to_idx))
        self.mc_to_idx = json.load(open(path_mc_to_idx))

        # load dataset
        path_data = Path(dataset_dir, f"{split}.csv")
        csv_ = pd.read_csv(path_data)
        print(f"read file : {path_data}, num_data : {len(csv_)}")
        self.topo = np.array(csv_["topo"])
        self.mc = np.array(csv_["mc"])
        self.num_conn = np.array(csv_["num_conn"])
        self.frags = np.array(csv_["frags"])
        self.selfies = np.array(csv_["selfies"])

        self.encoded_input, self.encoded_output = self.encoding(max_len)


    def encoding(self, max_len):
        # making encoded_input
        encoded_input = []
        for i, f in enumerate(self.frags):
            encoded_frags = [self.vocab_to_idx[v] for v in sf.split_selfies(f)]
            encoded = [self.mc_to_idx[self.mc[i]]] + \
                      [self.num_conn[i]] + \
                      [self.vocab_to_idx["[SOS]"]] + \
                      encoded_frags + \
                      [self.vocab_to_idx["[EOS]"]] + \
                      [self.vocab_to_idx["[PAD]"]] * (max_len - 4 - len(encoded_frags))
            encoded_input.append(encoded)

        # making encoded_output
        encoded_output = []
        for i, f in enumerate(self.selfies):
            encoded_selfies = [self.vocab_to_idx[v] for v in sf.split_selfies(f)]
            encoded = [self.vocab_to_idx["[SOS]"]] + \
                      [self.topo_to_idx[self.topo[i]]] + \
                      [self.mc_to_idx[self.mc[i]]] + \
                      encoded_selfies + \
                      [self.vocab_to_idx["[EOS]"]] + \
                      [self.vocab_to_idx["[PAD]"]] * (max_len - 4 - len(encoded_selfies))
            encoded_output.append(encoded)

        return encoded_input, encoded_output


    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        ret = dict()

        ret.update(
            {
                "topo" : self.topo[idx],
                "mc" : self.mc[idx],
                "frags" : self.frags[idx],
                "selfies" : self.selfies[idx],
                "encoded_input" : self.encoded_input[idx],
                "encoded_output" : self.encoded_output[idx],
            }
        )
        return ret

    @staticmethod
    def collate(batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        dict_batch["encoded_input"] = torch.LongTensor(dict_batch["encoded_input"])
        dict_batch["encoded_output"] = torch.LongTensor(dict_batch["encoded_output"])
        return dict_batch

