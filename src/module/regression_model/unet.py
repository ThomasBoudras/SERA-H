"""
U-Net Implementation.

This architecture is adapted from the Pytorch-UNet repository:
https://github.com/milesial/Pytorch-UNet

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation."
Medical image computing and computer-assisted intervention–MICCAI 2015.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.module.regression_model.components.unet_utils import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    """U-Net regression model that maps an input image to a single-channel output map.

    Standard encoder-decoder convolutional architecture with skip connections, adapted
    from the Pytorch-UNet implementation (see module docstring for reference).
    """

    def __init__(
        self,
        n_channels_in: int,
        bilinear: bool,
        out_activation: Optional[Any],
        encoder_widths: List[int],
        decoder_widths: List[int],
    ) -> None:
        """Build the U-Net encoder, decoder, and output convolution.

        Args:
            n_channels_in (int): Number of channels in the input images.
            bilinear (bool): If True, use bilinear upsampling in the decoder; otherwise use
                transposed convolutions.
            out_activation (Optional[Any]): Activation module applied to the output (e.g. ReLU),
                or None/"None"/"null" to disable it.
            encoder_widths (List[int]): Number of channels at each of the 5 encoder stages
                (from input resolution down to the bottleneck).
            decoder_widths (List[int]): Number of channels at each of the 4 decoder stages
                (from the bottleneck up to output resolution).
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels_in
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels_in, encoder_widths[0])
        self.down1 = Down(encoder_widths[0], encoder_widths[1])
        self.down2 = Down(encoder_widths[1], encoder_widths[2])
        self.down3 = Down(encoder_widths[2], encoder_widths[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(encoder_widths[3], encoder_widths[4] // factor)

        # Decoder
        self.up1 = Up(encoder_widths[4], decoder_widths[-1] // factor, bilinear)
        self.up2 = Up(decoder_widths[-1], decoder_widths[-2] // factor, bilinear)
        self.up3 = Up(decoder_widths[-2], decoder_widths[-3] // factor, bilinear)
        self.up4 = Up(decoder_widths[-3], decoder_widths[-4], bilinear)

        self.outc = OutConv(decoder_widths[-4], 1)

        self.out_activation = None
        if out_activation is not None:
            if (
                (out_activation == "None")
                or (out_activation is None)
                or (out_activation == "null")
            ):
                self.out_activation = None
            else:
                self.out_activation = out_activation

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Run the U-Net forward pass.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).
            labels (Optional[torch.Tensor]): Unused, kept for interface compatibility with
                other regression models that require targets.
            metadata (Optional[Dict[str, Any]]): Unused, kept for interface compatibility with
                other regression models that require metadata.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H, W), same spatial resolution as `inputs`.
        """
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)

        if self.out_activation is not None:
            output = self.out_activation(output)  # eg relu to avoid negative predictions
        # Now the output will have the same WxH as the input
        return output
