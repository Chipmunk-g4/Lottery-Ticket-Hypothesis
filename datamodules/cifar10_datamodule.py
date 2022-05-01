from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import numpy as np

class CIFAR10DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (45_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.indices_class: list = None
        self.images_all = []

    @property
    def num_classes(self) -> int:
        return 10

    def image_size(self):
        return (3,32,32)

    def mean_std(self):
        return self.mean, self.std

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir, train=True, transform=self.transform_train)
            testset = CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

        labels_all = [self.data_train[i][1] for i in range(len(self.data_train))]

        self.indices_class = [[] for c in range(self.num_classes)]
        for i, lab in enumerate(labels_all):
            self.indices_class[lab].append(i)

        self.images_all = [torch.unsqueeze(self.data_train[i][0], dim=0) for i in range(len(self.data_train))]
        self.images_all = torch.cat(self.images_all, dim=0)
    
    def get_class_image(self, c, n):
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
