from typing import Optional
from functools import partial

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from predictor.dataset import MOFDataset


class Datamodule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.dataset_dir = _config["dataset_dir"]
        self.batch_size = _config["batch_size"]
        self.num_workers = _config["num_workers"]
        self.max_len = _config["max_len"]
        self.tasks = [k for k, v in _config["loss_names"].items() if v >= 1]

    @property
    def dataset_cls(self):
        return MOFDataset

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            dataset_dir=self.dataset_dir,
            split="train",
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            dataset_dir=self.dataset_dir,
            split="val",
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            dataset_dir=self.dataset_dir,
            split="test",
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

        self.collate = partial(self.dataset_cls.collate, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=True,
            pin_memory=True,
        )
