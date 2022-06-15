from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from predictor.dataset import MOFDataset


class Datamodule(LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers=16):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size= batch_size
        self.num_workers = num_workers

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

    def setup(self, stage=None):
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
                          collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate)
