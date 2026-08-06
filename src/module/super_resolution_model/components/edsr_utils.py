"""Building blocks for the EDSR super-resolution architecture (convolutions, residual blocks,
and pixel-shuffle upsampler).
"""

import math
from typing import Any, Callable, List, Union

import torch
import torch.nn as nn


def conv2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Creates a 2D convolution with "same" padding derived from the kernel size.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the (square) convolution kernel.
        bias: Whether to include a learnable bias.

    Returns:
        A `nn.Conv2d` layer with padding set to `kernel_size // 2`.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    """Residual block used in the body of EDSR: two convolutions (with an optional activation
    and batch norm in between) whose output is scaled and added back to the block's input.
    """

    def __init__(
        self,
        conv: Callable[..., nn.Module],
        n_feats: int,
        kernel_size: int,
        bias: bool = True,
        bn: bool = False,
        act: nn.Module = nn.ReLU(True),
        res_scale: float = 1,
    ) -> None:
        """Initializes the residual block.

        Args:
            conv: Convolution constructor/factory, e.g. `conv2d`.
            n_feats: Number of feature channels (unchanged by this block).
            kernel_size: Kernel size passed to `conv`.
            bias: Whether the convolutions use a learnable bias.
            bn: If True, inserts a `BatchNorm2d` layer after each convolution.
            act: Activation module inserted after the first convolution.
            res_scale: Scaling factor applied to the residual branch before the skip connection.
        """
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the residual block.

        Args:
            x: Input tensor of shape (B, n_feats, H, W).

        Returns:
            Output tensor of the same shape as `x`.
        """
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    """Upsampling module for EDSR, built from convolutions and `PixelShuffle` layers.

    Supports scale factors of 1 (no spatial change, only convolutions), powers of two
    (via successive x2 pixel-shuffle stages), and 3 (via a single x3 pixel-shuffle stage).
    """

    def __init__(
        self,
        conv: Callable[..., nn.Module],
        scale: int,
        n_feats: int,
        bn: bool = False,
        act: Union[bool, str] = False,
        bias: bool = True,
    ) -> None:
        """Builds the sequence of layers implementing the requested upscaling factor.

        Args:
            conv: Convolution constructor/factory, e.g. `conv2d`.
            scale: Upscaling factor. Must be 1, 3, or a power of two.
            n_feats: Number of input feature channels.
            bn: If True, inserts a `BatchNorm2d` layer after each convolution.
            act: Activation to insert after each convolution: "relu", "prelu", or a falsy
                value (e.g. False) to skip activation.
            bias: Whether the convolutions use a learnable bias.

        Raises:
            NotImplementedError: If `scale` is not 1, 3, or a power of two.
        """
        m: List[Any] = []
        if scale == 1:
            for _ in range(2):
                m.append(conv(n_feats, n_feats, 3, bias))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
