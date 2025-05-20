import os
import random
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from matplotlib import pyplot as plt
from networkx.algorithms.operators.binary import intersection
from torch import nn, optim
from torch.cuda import device
from torch.nn import functional as F

from models.cnn_3d import TimeSeriesDiscriminator, RatEstimator3DCNN
from dataset.rat_dataset import RatDataset
from torch.utils import data
from torch.utils.data import Dataset


class ReduceDataset(Dataset):
    def __init__(self, dataset, data_size=20):
        super(ReduceDataset, self).__init__()
        self.dataset = dataset
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, _):
        index = random.randint(0, len(self.dataset) - 1)
        item = self.dataset[index]

        return item

class LightningRatDataset(L.LightningDataModule):
    def __init__(self, dataset, batch_size=1, num_workers=4):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set, self.val_set = None, None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_set, self.val_set = data.random_split(self.dataset, [len(self.dataset) - 20, 20])
            self.train_set = ReduceDataset(self.train_set, data_size=20)

    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class LightningRatModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.automatic_optimization = False

        self.g = RatEstimator3DCNN()

        self.d_b = TimeSeriesDiscriminator()
        self.d_h = TimeSeriesDiscriminator()

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()


    def forward(self, x):
        breathe_hat = self.g(x)

        return breathe_hat

    def training_step(self, batch, batch_idx):
        frames = batch["frames"]
        heart_rate = batch["heart_rate"]
        breathe = batch["breathe"]
        no_use_heart_rate = batch["no_use_heart_rate"]
        no_use_breathe = batch["no_use_breathe"]


        opt_d_b, opt_d_h, opt_g = self.optimizers()

        y_hat = self.g(frames)

        self.toggle_optimizer(opt_d_b)

        real_labels = torch.ones(breathe.size(0), 1, device=self.device)
        fake_labels = torch.zeros(y_hat[:, 0].size(0), 1, device=self.device)

        d_b_real = self.d_b(breathe)
        d_b_loss_real = self.bce_loss(d_b_real, real_labels)

        d_b_fake = self.d_b(y_hat[:, 0].detach())
        d_b_loss_fake = self.bce_loss(d_b_fake, fake_labels)

        d_b_loss = (d_b_loss_real + d_b_loss_fake) * 0.5
        self.log("train_d_b_real_loss", d_b_loss_real, prog_bar=True)
        self.log("train_d_b_fake_loss", d_b_loss_fake, prog_bar=True)
        self.log("train_d_b_loss", d_b_loss, prog_bar=True)

        self.manual_backward(d_b_loss)
        opt_d_b.step()
        opt_d_b.zero_grad()

        self.untoggle_optimizer(opt_d_b)
        self.toggle_optimizer(opt_d_h)

        real_labels = torch.ones(heart_rate.size(0), 1, device=self.device)
        fake_labels = torch.zeros(y_hat[:, 1].size(0), 1, device=self.device)

        d_h_real = self.d_h(heart_rate)
        d_h_loss_real = self.bce_loss(d_h_real, real_labels)

        d_h_fake = self.d_h(y_hat[:, 1].detach())
        d_h_loss_fake = self.bce_loss(d_h_fake, fake_labels)

        d_h_loss = (d_h_loss_real + d_h_loss_fake) * 0.5
        self.log("train_d_h_real_loss", d_h_loss_real, prog_bar=True)
        self.log("train_d_h_fake_loss", d_h_loss_fake, prog_bar=True)
        self.log("train_d_h_loss", d_h_loss, prog_bar=True)

        self.manual_backward(d_h_loss)
        opt_d_h.step()
        opt_d_h.zero_grad()

        self.untoggle_optimizer(opt_d_h)
        self.toggle_optimizer(opt_g)

        d_b_fake = self.d_b(y_hat[:, 0])
        d_h_fake = self.d_h(y_hat[:, 1])
        adv_b_loss = self.bce_loss(d_b_fake, torch.ones(y_hat.size(0), 1, device=self.device))
        adv_h_loss = self.bce_loss(d_h_fake, torch.ones(y_hat.size(0), 1, device=self.device))
        reg_b_loss = self.mse_loss(y_hat[:, 0] * no_use_breathe.int(), breathe * no_use_breathe.int())
        reg_h_loss = self.mse_loss(y_hat[:, 1] * no_use_heart_rate.int(), heart_rate  * no_use_heart_rate.int())
        lambda_adv = 0.1

        g_loss = reg_b_loss + reg_h_loss + lambda_adv * adv_b_loss + lambda_adv * adv_h_loss
        self.log("train_g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()

        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        frames = batch["frames"]
        heart_rate = batch["heart_rate"]
        breathe = batch["breathe"]
        no_use_heart_rate = batch["no_use_heart_rate"]
        no_use_breathe = batch["no_use_breathe"]

        y_hat = self.g(frames)

        breathe_loss = self.mse_loss(y_hat[:, 0] * no_use_breathe.int(), breathe * no_use_breathe.int())
        heart_loss = self.mse_loss(y_hat[:, 1] * no_use_heart_rate.int(), heart_rate * no_use_heart_rate.int())

        d_b_fake = self.d_b(y_hat[:, 0])
        d_h_fake = self.d_h(y_hat[:, 1])

        d_b_loss = self.bce_loss(d_b_fake, torch.ones(y_hat[:, 0].size(0), 1, device=self.device))
        d_h_loss = self.bce_loss(d_h_fake, torch.ones(y_hat[:, 1].size(0), 1, device=self.device))

        self.log("val_loss", breathe_loss + heart_loss)
        self.log("val_breathe_loss", breathe_loss, prog_bar=True)
        self.log("val_heart_loss", heart_loss, prog_bar=True)
        self.log("val_d_h_loss", d_h_loss)
        self.log("val_d_b_loss", d_b_loss)

    def configure_optimizers(self):
        opt_d_b = torch.optim.Adam(self.d_b.parameters(), lr=1e-3)
        opt_d_h = torch.optim.Adam(self.d_h.parameters(), lr=1e-3)
        opt_g = torch.optim.Adam(self.g.parameters(), lr=1e-3)

        return [opt_d_b, opt_d_h, opt_g], []


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    dataset =  RatDataset(
        data_root="../data/processed",
        session_id="20250329_150340",
        target_fps=30,
        input_frames=220, # Request 220 frames at 30 FPS
        output_frames=150 # Desired 150 frames at 30 FPS
    )

    lightning_module = LightningRatModel()

    lightning_data_module = LightningRatDataset(dataset, batch_size=1, num_workers=12)

    logger = TensorBoardLogger("tb_logs", name="3d-cnn-gan")
    checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        save_top_k=30,
        mode="max",
        filename="3d-cnn-gan-{epoch}-{val_loss:.5f}",
        dirpath="saves/3d-cnn-gan"
    )
    trainer = L.Trainer(max_epochs=1000, logger=logger, log_every_n_steps=1, callbacks=[checkpoint_callback], accelerator="gpu")
    trainer.fit(lightning_module, lightning_data_module)

