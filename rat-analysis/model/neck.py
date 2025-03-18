import math

import torch
from torch import nn
from torch.nn import functional as F

from model.resnet import ResNetBlock
from model.swin_transformer_3d import SwinTransformer3D


# class Swin3DNeck(nn.Module):
#     def __init__(self, embed_dim=96, frames=150, encoder_stride=4, patch_size=(4, 4, 4)):
#         super().__init__()
#
#         self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.seq = nn.Sequential(
#             ResNetBlock1D(embed_dim * 8, 256, kernel_size=3, padding=1),
#             ResNetBlock1D(256, 256, kernel_size=1, padding=0),
#             ResNetBlock1D(256, 256, kernel_size=3, padding=1),
#             ResNetBlock1D(256, 512, kernel_size=1, padding=0),
#             ResNetBlock1D(512, 128, kernel_size=1, padding=0),
#             ResNetBlock1D(128, 128, kernel_size=3, padding=1),
#             ResNetBlock1D(128, 256, kernel_size=1, padding=0),
#             ResNetBlock1D(256, 150, kernel_size=1, padding=0),
#             ResNetBlock1D(150, 150, kernel_size=3, padding=1),
#             ResNetBlock1D(150, 150, kernel_size=1, padding=0, activation=nn.Tanh()),
#         )
#         self.final_pool = nn.AdaptiveAvgPool1d(1)
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(
#                 in_channels=embed_dim*8,
#                 out_channels=1,
#                 kernel_size=(1, 32, 32),
#                 stride=(1, 32, 32),
#                 padding=(0, 0, 0)
#             ),
#             nn.Conv3d(1, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         mask = self.decoder(x)
#
#         x = self.pool(x).squeeze(-1).squeeze(-1)
#         x = self.seq(x)
#         x = self.final_pool(x)
#
#         return x.squeeze(-1), mask.squeeze(1)


class NeckResNet(nn.Module):
    def __init__(self, in_chans, out_chans, dimensions=1, first_pool=False, final_activation=nn.Tanh()):
        super().__init__()

        self.first_pool = first_pool
        if first_pool:
            self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.seq = nn.Sequential(
            ResNetBlock(dimensions, in_chans, 512, kernel_size=3, padding=1),
            ResNetBlock(dimensions, 512, 256, kernel_size=1, padding=0),
            ResNetBlock(dimensions, 256, 256, kernel_size=3, padding=1),
            ResNetBlock(dimensions, 256, 512, kernel_size=1, padding=0),
            ResNetBlock(dimensions, 512, 256, kernel_size=1, padding=0),
            ResNetBlock(dimensions, 256, 256, kernel_size=3, padding=1),
            ResNetBlock(dimensions, 256, out_chans, kernel_size=1, padding=0, activation=final_activation),
        )
        self.final_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if self.first_pool:
            x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.seq(x)
        x = self.final_pool(x)

        return x.squeeze(-1)


class BreatheNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(2),
            ResNetBlock(3, 64, 64, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            ResNetBlock(3, 64, 64, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            ResNetBlock(3, 64, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1), stride=2),
            ResNetBlock(3, 128, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            ResNetBlock(3, 128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=2),
            ResNetBlock(3, 256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ResNetBlock(3, 256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=2),
            ResNetBlock(3, 512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.AdaptiveAvgPool3d(1)
        )

        self.final = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * 14 * 14),
            nn.Sigmoid()
        )

        # self.swin3d = SwinTransformer3D(
        #     patch_size=(4, 4, 4),
        #     embed_dim=96,
        #     depths=[2, 2, 6, 2],
        #     num_heads=[3, 6, 12, 24],
        #     window_size=[8, 7, 7],
        #     mlp_ratio=4,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     drop_rate=0.,
        #     attn_drop_rate=0.,
        #     drop_path_rate=0.2,
        #     patch_norm=True
        # )
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.res_seq = nn.Sequential(
        #     ResNetBlock(1, 768, 768, kernel_size=3, padding=1),
        #     nn.MaxPool1d(2),
        #     ResNetBlock(1, 768, 512, kernel_size=3, padding=1),
        #     nn.MaxPool1d(2),
        #     ResNetBlock(1, 512, 256, kernel_size=3, padding=1),
        #     nn.MaxPool1d(2),
        #     ResNetBlock(1, 256, 150, kernel_size=1, padding=0, activation=nn.Tanh()),
        #     nn.AdaptiveAvgPool1d(1)
        # )


    def forward(self, x):
        x = self.seq(x).view(-1, 512)
        x = self.final(x).view(-1, 2, 14, 14)

        # x = self.swin3d(x)
        # x = self.pool(x).squeeze(-1).squeeze(-1)
        # x = self.res_seq(x)

        # return x.squeeze(-1)
        return x



if __name__ == "__main__":
    model = BreatheNeck()

    x = torch.rand(2, 3, 151, 224, 224)
    y = model(x)

    print()
