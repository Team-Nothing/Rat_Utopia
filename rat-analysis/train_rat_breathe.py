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

from model.BreatheEstimator import BreatheEstimator, BreatheEstimator3DCNN, BreathingDiscriminator, \
    StrongBreathingDiscriminator
from model.neck import NeckResNet, BreatheNeck
from dataset.rat_dataset import BreathDataset
from torch.utils import data

from model.u_net import UNet
import train_rat_mask


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
    def __init__(self, mask_model):
        super().__init__()

        self.automatic_optimization = False
        self.rat_mask = mask_model

        self.generator = BreatheEstimator3DCNN()
        self.discriminator = BreathingDiscriminator()

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        for param in self.rat_mask.parameters():
            param.requires_grad = False


    def forward(self, x):
        B, T, C, H, W = x.shape

        mask_hat = x.view(B * T, C, H, W)
        mask_hat = self.rat_mask(mask_hat)
        mask_hat = mask_hat.view(B, T, 1, H, W)

        breathe_hat = self.generator(x, mask_hat[:, 1:])

        return breathe_hat

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        opt_disc, opt_gen = self.optimizers()

        y_hat = self(x)

        self.toggle_optimizer(opt_disc)

        real_labels = torch.ones(y.size(0), 1, device=self.device)
        fake_labels = torch.zeros(y_hat.size(0), 1, device=self.device)

        d_real = self.discriminator(y)
        d_loss_real = self.bce_loss(d_real, real_labels)

        d_fake = self.discriminator(y_hat.detach())
        d_loss_fake = self.bce_loss(d_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        self.log("train_d_real_loss", d_loss_real, prog_bar=True)
        self.log("train_d_fake_loss", d_loss_fake, prog_bar=True)
        self.log("train_d_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        opt_disc.step()
        opt_disc.zero_grad()

        self.untoggle_optimizer(opt_disc)
        self.toggle_optimizer(opt_gen)

        d_fake = self.discriminator(y_hat)
        adv_loss = self.bce_loss(d_fake, torch.ones(y_hat.size(0), 1, device=self.device))
        reg_loss = self.mse_loss(y_hat, y)
        lambda_adv = 0.1

        g_loss = reg_loss + lambda_adv * adv_loss
        self.log("train_g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        opt_gen.step()
        opt_gen.zero_grad()

        self.untoggle_optimizer(opt_gen)

    def validation_step(self, batch, batch_idx):
        x, _, breathe = batch

        breathe_hat = self(x)
        breathe_loss = self.mse_loss(breathe_hat, breathe)
        d_fake = self.discriminator(breathe_hat)
        d_loss = self.bce_loss(d_fake, torch.ones(breathe_hat.size(0), 1, device=self.device))

        self.log("val_loss", (breathe_loss + d_loss) * 0.5, prog_bar=True)
        self.log("val_breathe_loss", breathe_loss, prog_bar=True)


    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=1e-3)

        return [opt_disc, opt_gen], []


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    dataset = BreathDataset(data_path="data/records")

    mask_model = train_rat_mask.LightningRatModel.load_from_checkpoint("saves/u-net-mask/u-net-mask-v2-epoch=10-val_loss=0.00811.ckpt").rat_mask

    lightning_module = LightningRatModel(mask_model)

    lightning_data_module = LightningRatDataset(dataset, batch_size=3, num_workers=18)

    logger = TensorBoardLogger("tb_logs", name="u-net-mask-breathe")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="u_net_mask-3d_cnn-gan-{epoch}-{val_loss:.5f}",
        dirpath="saves/u-net-mask-breathe"
    )
    trainer = L.Trainer(max_epochs=10000, logger=logger, log_every_n_steps=10, callbacks=[checkpoint_callback])
    trainer.fit(lightning_module, lightning_data_module)

