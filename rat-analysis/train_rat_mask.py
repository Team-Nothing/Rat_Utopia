import os
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from matplotlib import pyplot as plt
from networkx.algorithms.operators.binary import intersection
from torch import nn, optim
from torch.nn import functional as F

from model.neck import NeckResNet, BreatheNeck
from dataset.rat_dataset import BreathDataset
from torch.utils import data

from model.u_net import UNet


class LightningRatDataset(L.LightningDataModule):
    def __init__(self, dataset, batch_size=1, num_workers=4):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set, self.val_set = None, None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_set, self.val_set = data.random_split(self.dataset, [int(len(self.dataset) * 0.8), len(self.dataset) - int(len(self.dataset) * 0.8)])

    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class LightningRatModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.rat_mask = UNet(in_chans=3, out_chans=1)
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def dice_loss(y_hat, y, smooth=1.):
        y_hat = F.sigmoid(y_hat)
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        intersection = (y_hat * y).sum()
        dice = (2. * intersection + smooth) / (y_hat.sum() + y.sum() + smooth)

        return 1 - dice


    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        mask_hat = self.rat_mask(x)
        mask_hat = mask_hat.view(B, T, H, W)

        return mask_hat

    def training_step(self, batch, batch_idx):
        x, mask, _ = batch
        B, T, _, _, _ = x.shape

        mask_hat = self(x)
        mask_loss = self.mse_loss(mask_hat, mask.repeat(B, T, 1, 1))

        self.log("train_loss", mask_loss, prog_bar=True)

        return mask_loss


    def validation_step(self, batch, batch_idx):
        x, mask, _ = batch
        B, T, _, _, _= x.shape

        mask_hat = self(x)
        mask_loss = self.mse_loss(mask_hat, mask.repeat(B, T, 1, 1))

        self.log("val_loss", mask_loss, prog_bar=True)


    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.parameters(), lr=1e-3)

        return [opt_g], []


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    dataset = BreathDataset(data_path="data/records")

    lightning_module = LightningRatModel()
    lightning_data_module = LightningRatDataset(dataset, batch_size=1, num_workers=4)

    logger = TensorBoardLogger("tb_logs", name="u-net-mask")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="u-net-mask-v2-{epoch}-{val_loss:.5f}",
        dirpath="saves/u-net-mask"
    )
    trainer = L.Trainer(max_epochs=2000, logger=logger, log_every_n_steps=1, callbacks=[checkpoint_callback])
    trainer.fit(lightning_module, lightning_data_module)

