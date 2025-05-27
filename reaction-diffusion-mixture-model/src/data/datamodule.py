import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
import os

from data.dataset import HeatDataset


class HeatDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.save_hyperparameters()
        self.hyperparams = hyperparams
        self.batch_size = hyperparams['batch_size']
        self.data_dir = hyperparams['data_dir']
        self.num_samples = hyperparams['num_samples']
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # load data from file
        print("Loading data from file...")

        train_data = np.load(f"{self.data_dir}/train_data.npy")
        train_mask = np.load(f"{self.data_dir}/train_mask.npy")
        train_time = np.load(f"{self.data_dir}/train_time.npy")

        val_data = np.load(f"{self.data_dir}/val_data.npy")
        val_mask = np.load(f"{self.data_dir}/val_mask.npy")
        val_time = np.load(f"{self.data_dir}/val_time.npy")

        test_data = np.load(f"{self.data_dir}/test_data.npy")
        test_mask = np.load(f"{self.data_dir}/test_mask.npy")
        test_time = np.load(f"{self.data_dir}/test_time.npy")

        self.train_dataset = HeatDataset(train_data, train_mask, train_time)
        self.val_dataset = HeatDataset(val_data, val_mask, val_time)
        self.test_dataset = HeatDataset(test_data, test_mask, test_time)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=False, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.num_samples, num_workers=os.cpu_count(), shuffle=False, pin_memory=True, persistent_workers=True)

