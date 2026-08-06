"""Building blocks (double convolution, down/up-sampling) used by the U-Net architecture."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ) -> None:
        """Build the two convolution (+ BatchNorm + ReLU) layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (Optional[int]): Number of channels between the two convolutions.
                Defaults to `out_channels` if not specified.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the max-pooling downsampling followed by a `DoubleConv`.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downscale by a factor of 2 and apply the double convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H/2, W/2).
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        """Build the upsampling operation followed by a `DoubleConv`.

        Args:
            in_channels (int): Number of input channels (of the tensor to upsample).
            out_channels (int): Number of output channels.
            bilinear (bool): If True, use bilinear upsampling followed by a channel-reducing
                `DoubleConv`; otherwise use a transposed convolution.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample `x1`, pad it to match `x2`'s spatial size, concatenate, and convolve.

        Args:
            x1 (torch.Tensor): Lower-resolution tensor to upsample, shape (B, in_channels, H, W).
            x2 (torch.Tensor): Skip-connection tensor from the encoder, used to determine the
                target spatial size and concatenated along the channel dimension.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H2, W2), where (H2, W2) is
                the spatial size of `x2`.
        """
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Final 1x1 convolution mapping decoder features to the output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the 1x1 output convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 1x1 convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        return self.conv(x)
