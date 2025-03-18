import json
import math
import os
import random
import shutil
import time

import numpy as np

import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import yscale
from torch.utils.data import Dataset
from torch import nn
from torchvision import transforms as VT
from torchvision.transforms import functional as VTF
from torchaudio import transforms as AT
from torch.nn import functional as F


class ScaleTransform(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, img):
        original_size = img.shape[-2:]
        new_size = [int(original_size[0] * self.scale_factor), int(original_size[1] * self.scale_factor)]

        img_resized = VTF.resize(img, new_size)
        img_fixed = VTF.center_crop(img_resized, original_size)  # Ensures same final shape

        return img_fixed


class AffineTransform(nn.Module):
    def __init__(self, rotate_angle, scale_factor, shift_range):
        """
            Args:
                rotate_angle (tuple): (min_angle, max_angle) in degrees.
                scale_factor (tuple): (min_scale, max_scale).
                shift_range (tuple): (min_shift, max_shift) in normalized coordinates.
        """
        super().__init__()
        self.rotate_angle = rotate_angle
        self.scale_factor = scale_factor
        self.shift_range = shift_range

    def forward(self, *x):

        out = []
        rotate_angle = torch.FloatTensor(1).uniform_(*self.rotate_angle).item()
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_factor).item()
        shift_x = torch.FloatTensor(1).uniform_(*self.shift_range).item()
        shift_y = torch.FloatTensor(1).uniform_(*self.shift_range).item()

        angle_rad = torch.deg2rad(torch.tensor(rotate_angle))
        cos_val = torch.cos(angle_rad) * scale_factor
        sin_val = torch.sin(angle_rad) * scale_factor
        theta = torch.tensor([
            [cos_val, -sin_val, shift_x],
            [sin_val, cos_val, shift_y]
        ], dtype=torch.float).unsqueeze(0)

        for x_ in x:
            T, C, H, W = x_.shape

            theta_ = theta.repeat(T, 1, 1)

            grid = F.affine_grid(theta_, x_.size(), align_corners=False)
            x_ = F.grid_sample(x_, grid, align_corners=False)

            out.append(x_)

        if len(out) == 1:
            return out[0]

        return tuple(out)


class SpeedTransform(nn.Module):
    def __init__(self, speed_factor, out_frames):
        super().__init__()
        self.speed_factor = speed_factor
        self.out_frames = out_frames

    def forward(self, x, y=None):
        T, C, H, W = x.shape

        speed_factor = torch.FloatTensor(1).uniform_(*self.speed_factor).item()
        T_new = int(round(T * speed_factor))
        T_new = max(T_new, 1)

        x = x.permute(1, 0, 2, 3)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(T_new, H, W),
            mode='trilinear',
            align_corners=True
        ).squeeze(0)
        x = x.permute(1, 0, 2, 3)

        if T_new <= self.out_frames:
            num_to_pad = self.out_frames - T_new
            last_frame = x[-1, :, :, :]

            padding = last_frame.repeat(num_to_pad, 1, 1, 1)
            x = torch.cat([x, padding], dim=0)
        else:
            start = random.randint(0, T_new - self.out_frames)
            x = x[start:start + self.out_frames, :, :, :]

        if y is not None:
            y = F.interpolate(
                y.unsqueeze(0).unsqueeze(0),
                size=(T_new),
                mode='linear',
                align_corners=True
            ).squeeze(0).squeeze(0)
            if T_new <= self.out_frames:
                num_to_pad = self.out_frames - T_new
                last_frame = y[-1]
                padding = last_frame.repeat(num_to_pad)
                y = torch.cat([y, padding], dim=0)
            else:
                start = random.randint(0, T_new - self.out_frames)
                y = y[start:start + self.out_frames]


            return x, y

        return x


class BreathDataset(Dataset):
    def __init__(self,
                 data_path="../data/records",
                 frames=750,
                 speed_factor=(0.8, 1.25),
                 scale_factor=(0.5, 1.5),
                 rotate_angle=(-180, 180),
                 shift_range=(-0.2, 0.2),
                 out_frames=151,
                 y_scale=10):
        self.data_path = data_path
        self.frames = frames
        self.out_frames = out_frames
        self.speed_factor = speed_factor
        self.scale_factor = scale_factor
        self.rotate_angle = rotate_angle
        self.shift_range = shift_range
        self.y_scale = y_scale

        self.affine_transform = AffineTransform(rotate_angle, scale_factor, shift_range)
        self.speed_transform = SpeedTransform(speed_factor, self.out_frames)
        self.y_transform = AT.Spectrogram(n_fft=26, win_length=22, power=None)

        self.data = []
        self.frame_data = {}
        self.breathe_data = {}
        self.mask_data = {}

        mask_transform = VT.Compose([
            VT.Resize((224, 448)),
            VT.ToTensor()
        ])

        for record in os.listdir(data_path):
            record_dir = os.path.join(data_path, record)
            if not os.path.isdir(record_dir):
                continue

            with open(os.path.join(record_dir, "config.json")) as f:
                config = json.load(f)

            total_frame = config["train"]["total-frames"]
            no_use_ranges = config["train"]["no-use"]

            self.frame_data[os.path.join(record_dir, "frames.npy")] = torch.from_numpy(np.load(os.path.join(record_dir, "frames.npy"))).float()
            r_smooth = torch.from_numpy(np.load(os.path.join(record_dir,"r_smoothed.npy"))).float()
            self.breathe_data[os.path.join(record_dir, "r_smoothed.npy")] = r_smooth

            mask_image = Image.open(os.path.join(record_dir, "mask.png"))
            self.mask_data[os.path.join(record_dir, "mask.png")] = mask_transform(mask_image)

            i = 0
            while i < total_frame:
                in_no_use_range = False
                for start, end in no_use_ranges:
                    if start <= i <= end or start <= i + self.frames - 1 <= end:
                        i = end + 1
                        in_no_use_range = True
                        break

                if in_no_use_range:
                    continue

                self.data.append((
                    os.path.join(record_dir, "frames.npy"),
                    os.path.join(record_dir, "r_smoothed.npy"),
                    os.path.join(record_dir, "mask.png"),
                    i, i + self.frames))
                i += 1

    def __len__(self):
        return len(self.data) // 500

    def __getitem__(self, idx):
        idx = random.randint(idx * 500, (idx + 1) * 500 - 1)
        frame_path, rect_rgb_mean_path, mask_path, start, end = self.data[idx]
        left_right = random.randint(0, 1)

        x = self.frame_data[frame_path][start:end:4, :, :, 224 * left_right: 224 * (left_right + 1)]
        mask = self.mask_data[mask_path][:, :, 224 * left_right: 224 * (left_right + 1)]
        y = self.breathe_data[rect_rgb_mean_path][start:end:4]

        x, mask = self.affine_transform(x, mask.unsqueeze(0))
        x, y = self.speed_transform(x, y)
        y = y[1:] - y[:-1]

        # plt.plot(y.cpu().detach().numpy(), label="y")
        # plt.legend()
        # plt.show()

        # real_part = y.real
        # imag_part = y.imag
        #
        # # Convert to dB scale for visualization (if needed)
        # real_part_db = 20 * torch.log10(torch.abs(real_part) + 1e-6)
        # imag_part_db = 20 * torch.log10(torch.abs(imag_part) + 1e-6)
        #
        # # Plot both real and imaginary parts
        # fig, ax = plt.subplots(2, 1, figsize=(4, 8), sharex=True)
        #
        # # Plot real part
        # ax[0].imshow(real_part_db.cpu().numpy(), aspect="auto", cmap="inferno", origin="lower")
        # ax[0].set_title("Real Part (Magnitude in dB)")
        # ax[0].set_ylabel("Frequency Bins")
        #
        # # Plot imaginary part
        # ax[1].imshow(imag_part_db.cpu().numpy(), aspect="auto", cmap="inferno", origin="lower")
        # ax[1].set_title("Imaginary Part (Magnitude in dB)")
        # ax[1].set_xlabel("Time Steps")
        # ax[1].set_ylabel("Frequency Bins")
        #
        # plt.tight_layout()
        # plt.show()

        # for _x, _mask in zip(x, mask):
        #     plt.imshow(_x[0].numpy())
        #     plt.imshow(_mask[0].numpy(), alpha=0.5)
        #     plt.show()
        #     print()

        return x, mask.squeeze(0), y


if __name__ == "__main__":
    dataset = BreathDataset()

    data = [dataset[i] for i in range(10)]

    print()
