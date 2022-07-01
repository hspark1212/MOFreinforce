import torch
import json
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MOFDataset(Dataset):
    def __init__(self, dataset_dir, split, target):
        assert split in ["train", "test", "val"]
        path_dict_mof = Path(dataset_dir, f"{target}_dict_mof_{split}.json")
        print(f"read file {path_dict_mof}")
        self.dict_mof = json.load(open(path_dict_mof, "r"))
        self.mof_name = list(self.dict_mof.keys())

    def __len__(self):
        return len(self.dict_mof)

    def __getitem__(self, idx):
        return self.dict_mof[self.mof_name[idx]]

    @staticmethod
    def collate(batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # target
        dict_batch["target"] = torch.FloatTensor(dict_batch["target"])
        # mc
        dict_batch["mc"] = torch.LongTensor(dict_batch["mc"])
        # topo
        dict_batch["topo"] = torch.LongTensor(dict_batch["topo"])
        # ol (make pad)
        dict_batch_ol = [torch.LongTensor(ol) for ol in dict_batch["ol"]]
        dict_batch_ol_pad = pad_sequence(dict_batch_ol, batch_first=True, padding_value=0)
        dict_batch["ol_pad"] = dict_batch_ol_pad
        dict_batch["ol_len"] = [len(ol) for ol in dict_batch["ol"]]
        return dict_batch
