from torchvision import models
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation: any = nn.Identity()):
        super().__init__()

        if dimensions == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dimensions == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif dimensions == 3:
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            raise ValueError("Only 1, 2, or 3 dimensions are supported")

        if in_channels != out_channels:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.sequential = nn.Sequential(
            conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            bn(out_channels),
            nn.GELU(),
            conv(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            bn(out_channels),
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(self.shortcut(x) + self.sequential(x))



class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resnet = models.resnet50(pretrained=False)

        self.gru = nn.GRU(1000, hidden_size=256, num_layers=3, batch_first=True)
        self.fc = nn.Linear(256, 1)
        self.ac = nn.ReLU()

    def forward(self, x, hidden):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.resnet(x)
        x = x.view(B, T, -1)
        x, hidden = self.gru(x, hidden)
        x = self.ac(x)
        x = self.fc(x)

        return x, hidden

