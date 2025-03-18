import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchaudio import transforms as AT

from model.neck import NeckResNet, BreatheNeck
from model.swin_transformer_3d import SwinTransformer3D
from model.u_net import UNet
from train_rat_breathe import LightningRatModel
import train_rat_mask

import matplotlib.pyplot as plt

FRAMES = 600
RECORD_PATH = "data/records"
RECORD_ID = "20241123_121043"
CKPT_PATH = "saves/u-net-mask-breathe/u_net_mask-3d_cnn-gan-epoch=1232-val_loss=0.13274.ckpt"
LEFT_RIGHT = 0

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    mask_model = train_rat_mask.LightningRatModel.load_from_checkpoint("saves/u-net-mask/u-net-mask-v2-epoch=10-val_loss=0.00811.ckpt").rat_mask
    lightning_module = LightningRatModel.load_from_checkpoint(CKPT_PATH, mask_model=mask_model)
    lightning_module.cpu().eval()

    frames = torch.from_numpy(np.load(f"{RECORD_PATH}/{RECORD_ID}/frames.npy")).float()
    real_y = torch.from_numpy(np.load("data/records/20241123_121043/r_smoothed.npy")).float()

    for i in range(1000, len(frames) - FRAMES - 1, FRAMES + 1):
        x = frames[i:i + FRAMES + 1][::4]
        y = real_y[i:i + FRAMES][::4]
        T, C, H, W = x.shape
        x = x[..., (W // 2) * LEFT_RIGHT:(W // 2) * (LEFT_RIGHT + 1)]

        breathe_hat = lightning_module(x.unsqueeze(0).to(lightning_module.device))
        breathe_hat = breathe_hat.squeeze(0)

        y = y[1:] - y[:-1]

        plt.plot(breathe_hat.cpu().detach().numpy(), label="y_hat")
        plt.plot(y.cpu().detach().numpy(), label="y")
        plt.legend()
        plt.show()

        print()
        # output = torch.cat([x.permute(1, 0, 2, 3), mask_hat.repeat(1, 3, 1, 1)], dim=-1)
        #
        # fps = 30  # Frames per second
        # video_filename = "output.mp4"
        # height, width = output.shape[2], output.shape[3]  # Assuming shape (151, 1, H, W)
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #
        # video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height), isColor=False)
        #
        # for i in range(151):
        #     frame = output[i, 0].cpu().detach().numpy()  # Convert tensor to numpy
        #     frame = (frame * 255).astype(np.uint8)  # Normalize to 0-255
        #     video_writer.write(frame)  # Write frame to video
        #
        # video_writer.release()
        print()
        # y = y[1:] - y[:-1]



        print()




