import torch
import json
from pathlib import Path
from torch.utils.data import Dataset


class MOFDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 tasks=None,
                 ):
        assert split in ["train", "test", "val"]

        # load dict_mof
        path_dict_mof = Path(dataset_dir, f"dict_mof.json")
        print(f"read file : {path_dict_mof}")
        self.dict_mof = json.load(open(path_dict_mof, "r"))

        # load tasks
        if tasks is None:
            tasks = []
        print(f"tasks : {tasks}")
        for task in tasks:
            if task in ["trc", "vfr"]:
                path_dict_target = Path(dataset_dir, f"{task}_{split}.json")
                print(f"read file : {path_dict_target}")
                self.dict_target = json.load(open(path_dict_target, "r"))
            else:
                assert Exception, print("check loss_names in config_predictor.py")


        self.mof_name = list(self.dict_target.keys())


    def __len__(self):
        return len(self.mof_name)

    def __getitem__(self, idx):
        ret = dict()
        mof_name = self.mof_name[idx]

        ret.update(
            {
                "mof_name": mof_name,
                "target": self.dict_target[mof_name],
            }

        )
        ret.update(self.dict_mof[mof_name])
        return ret

    @staticmethod
    def collate(batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # target
        dict_batch["target"] = torch.FloatTensor(dict_batch["target"])
        # mc (idx)
        dict_batch["mc"] = torch.LongTensor(dict_batch["mc"])
        # topo (idx)
        dict_batch["topo"] = torch.LongTensor(dict_batch["topo"])
        # ol (selfies)
        dict_batch["ol"] = torch.LongTensor(
            [ol + [0] * (100 - len(ol)) if len(ol) < 100 else ol[:100] for ol in dict_batch["ol"]]
        )
        return dict_batch
