from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from generator.dataset import GeneratorDataset


class GeneratorDatamodule(LightningDataModule):
    def __init__(self, _config):
        super(GeneratorDatamodule, self).__init__()
        self.dataset_dir = _config["dataset_dir"]
        self.batch_size = _config["batch_size"]
        self.num_workers = _config["num_workers"]
        self.max_len = _config["max_len"]
        self.path_vocab = _config["path_vocab"]
        self.path_topo_to_idx = _config["path_topo_to_idx"]
        self.path_mc_to_idx = _config["path_mc_to_idx"]

    @property
    def dataset_cls(self):
        return GeneratorDataset

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            dataset_dir=self.dataset_dir,
            path_vocab=self.path_vocab,
            path_topo_to_idx=self.path_topo_to_idx,
            path_mc_to_idx=self.path_mc_to_idx,
            split="train",
            max_len=self.max_len,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            dataset_dir=self.dataset_dir,
            path_vocab=self.path_vocab,
            path_topo_to_idx=self.path_topo_to_idx,
            path_mc_to_idx=self.path_mc_to_idx,
            split="val",
            max_len=self.max_len,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            dataset_dir=self.dataset_dir,
            path_vocab=self.path_vocab,
            path_topo_to_idx=self.path_topo_to_idx,
            path_mc_to_idx=self.path_mc_to_idx,
            split="test",
            max_len=self.max_len,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

        self.collate = self.dataset_cls.collate


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate,
                          shuffle=True, ##
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate,
                          shuffle=False, ##
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate,
                          shuffle=False, ##
                          pin_memory=True)



