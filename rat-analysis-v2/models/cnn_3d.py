from pyexpat import features
from unittest.mock import inplace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class RatEstimator3DCNN(nn.Module):
    def __init__(self, y_dim=2):
        """
        A 3D CNN that estimates the breathing signal from an input video.

        We assume the input video x is of shape (B, T, C, H, W) with C=3. We first
        compute the per-frame differences (yielding T-1 frames) and concatenate these
        with the original frames (from 1 to T) so that each “time step” has 6 channels.

        The network then permutes the data into the 3D CNN standard format:
            (B, channels, T, H, W)
        and applies several 3D convolution blocks that downsample spatially only.

        Finally, we average pool over the spatial dimensions and apply a linear layer
        (applied at each time step) to produce a breathing value per time step.

        Args:
            feature_channels (int): Number of channels produced by the last 3D conv block.
        """
        super(RatEstimator3DCNN, self).__init__()
        # The network expects 6 channels (3 original + 3 difference)
        self.conv1 = nn.Conv3d(
            in_channels=6, out_channels=32,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(
            in_channels=64, out_channels=128,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(
            in_channels=128, out_channels=256,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn4 = nn.BatchNorm3d(256)

        self.conv5 = nn.Conv3d(
            in_channels=256, out_channels=512,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn5 = nn.BatchNorm3d(512)

        self.relu = nn.ReLU(inplace=True)
        # Final fully-connected layer applied to each time step independently
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, y_dim)
        )

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x (Tensor): Video tensor of shape (B, T, C, H, W). We assume T>=2.
            mask (Tensor, optional): Mask tensor of shape (B, T-1, H, W). If provided,
                we enhance the frames with: x_enhanced = x * (0.5 + sigmoid(mask)).
                (Note: We apply the mask to both the raw frames and their differences.)

        Returns:
            Tensor: Predicted breathing signal of shape (B, T-1)
        """
        # Compute the “original” frames from time 1 onward.
        # (We drop the first frame so that we can compute differences.)
        x1 = x[:, 1:, :, :, :]  # Shape: (B, T-1, C, H, W)
        # Compute temporal differences: frame difference between consecutive frames.
        x_diff = x1 - x[:, :-1, :, :, :]  # Shape: (B, T-1, C, H, W)

        # If a mask is provided, use it to enhance both the raw and difference signals.
        if mask is not None:
            # mask is assumed to have shape (B, T-1, H, W); add a channel dimension.
            mask_enhance = 0.1 + torch.sigmoid(mask)  # Shape: (B, T-1, 1, H, W)
            x1 = x1 * mask_enhance
            x_diff = x_diff * mask_enhance

        # Concatenate the raw frames and their differences along the channel dimension.
        # Now each time step has 6 channels.
        x_cat = torch.cat([x1, x_diff], dim=2)  # Shape: (B, T-1, 6, H, W)
        # Rearrange to the 3D CNN input format: (B, channels, T, H, W)
        x_cat = x_cat.permute(0, 2, 1, 3, 4)  # Now shape: (B, 6, T-1, H, W)

        # Pass through the 3D convolutional blocks.
        x_conv = self.conv1(x_cat)
        x_conv = self.bn1(x_conv)
        x_conv = self.relu(x_conv)

        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = self.relu(x_conv)

        x_conv = self.conv3(x_conv)
        x_conv = self.bn3(x_conv)
        x_conv = self.relu(x_conv)

        x_conv = self.conv4(x_conv)
        x_conv = self.bn4(x_conv)
        x_conv = self.relu(x_conv)

        x_conv = self.conv5(x_conv)
        x_conv = self.bn5(x_conv)
        x_conv = self.relu(x_conv)
        # At this point, x_conv has shape: (B, feature_channels, T-1, H', W')
        # where the temporal dimension (T-1) is preserved because we used stride=1 in time.

        # Global average pooling over the spatial dimensions.
        x_pooled = x_conv.mean(dim=[3, 4])  # New shape: (B, feature_channels, T-1)

        # Rearrange so that the time dimension comes second.
        x_pooled = x_pooled.permute(0, 2, 1)  # Shape: (B, T-1, feature_channels)

        # Apply a fully-connected layer to each time step to produce a scalar.
        out = self.fc(x_pooled)  # Shape: (B, T-1, y_dim)
        out = out.transpose(1, 2)  # Shape: (B, u_dim, T-1)
        return out


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self):
        super(TimeSeriesDiscriminator, self).__init__()
        # A simple 1D CNN that takes a breathing signal (B, T) and outputs a probability.
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Breathing signal of shape (B, T)
        Returns:
            Tensor: Discriminator output probability (B, 1)
        """
        # Add channel dimension → (B, 1, T)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class StrongBreathingDiscriminator(nn.Module):
    def __init__(self):
        """
        A stronger discriminator that processes 1D breathing signals.
        Input: breathing signal of shape (B, T)
        Output: probability score (B, 1)
        """
        super(StrongBreathingDiscriminator, self).__init__()
        # We assume the input breathing signal is a 1D sequence.
        # First, we add a channel dimension so that input becomes (B, 1, T).
        # We then apply several spectral normalized Conv1d layers to increase capacity.
        self.conv1 = spectral_norm(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.conv5 = spectral_norm(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))

        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Instead of manually computing the sequence length after several strides,
        # we use an adaptive pooling layer.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Breathing signal of shape (B, T)
        Returns:
            Tensor: Probability score for each sample of shape (B, 1)
        """
        # Add channel dimension: (B, 1, T)
        x = x.unsqueeze(1)
        # Pass through the convolutional layers with dropout and activation.
        x = self.leaky_relu(self.conv1(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.conv3(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.conv4(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.conv5(x))
        x = self.dropout(x)

        # Adaptive pooling to collapse the temporal dimension.
        x = self.adaptive_pool(x)  # shape: (B, 512, 1)
        x = x.view(x.size(0), -1)  # shape: (B, 512)
        x = self.fc(x)  # shape: (B, 1)
        x = torch.sigmoid(x)  # output probability
        return x


if __name__ == '__main__':
    model = RatEstimator3DCNN()
    x = torch.randn(2, 151, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print()
