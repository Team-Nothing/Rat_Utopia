from pyexpat import features

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class BreatheEstimator(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128):
        """

        Args:
            feature_dim: dimension of the spatial features after the CNN encoder.
            hidden_dim: LSTM hidden dimension (note: we use a bidirectional LSTM).

        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 224 * 224 -> 112 * 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112 * 112 -> 56 * 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56 * 56 -> 28 * 28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28 * 28 -> 14 * 14

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, 3, bidirectional=True, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, mask=None, hidden=None):
        """

        Args:
            x: video tensor of shape (B, T, C, H, W)
            mask: mask tensor of shape (B, T, 1, H, W)

        Returns: breathe prediction tensor of shape (B, T)

        """

        x1 = x[:, 1:, :, :, :]
        x_diff = x1 - x[:, :-1, :, :, :]

        if mask is not None:
            mask_enhance = 0.5 + F.sigmoid(mask)
            x1 = x1 * mask_enhance
            x_diff = x_diff * mask_enhance

        x = torch.cat([x1, x_diff], dim=2)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B * T, -1)
        x = self.fc(x)

        x = x.view(B, T, -1)

        x, hidden = self.lstm(x, hidden)

        x = self.out_fc(x)
        x = F.tanh(x.squeeze(-1))

        return x, hidden


class BreatheEstimator3DCNN(nn.Module):
    def __init__(self, feature_channels=128):
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
        super(BreatheEstimator3DCNN, self).__init__()
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
            in_channels=64, out_channels=feature_channels,
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(feature_channels)

        self.relu = nn.ReLU(inplace=True)
        # Final fully-connected layer applied to each time step independently
        self.fc = nn.Sequential(
            nn.Linear(feature_channels, feature_channels),
            nn.Linear(feature_channels, feature_channels),
            nn.Linear(feature_channels, 1)
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
        # At this point, x_conv has shape: (B, feature_channels, T-1, H', W')
        # where the temporal dimension (T-1) is preserved because we used stride=1 in time.

        # Global average pooling over the spatial dimensions.
        x_pooled = x_conv.mean(dim=[3, 4])  # New shape: (B, feature_channels, T-1)

        # Rearrange so that the time dimension comes second.
        x_pooled = x_pooled.permute(0, 2, 1)  # Shape: (B, T-1, feature_channels)

        # Apply a fully-connected layer to each time step to produce a scalar.
        out = self.fc(x_pooled)  # Shape: (B, T-1, 1)
        out = out.squeeze(-1)  # Shape: (B, T-1)
        return out


class BreathingDiscriminator(nn.Module):
    def __init__(self):
        super(BreathingDiscriminator, self).__init__()
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
    model = BreatheEstimator()
    x = torch.randn(2, 151, 3, 224, 224)
    mask = torch.randn(2, 150, 224, 224)
    y = model(x, mask)
    print(y.shape)
    print(y)
    print(y.shape)
    print()
