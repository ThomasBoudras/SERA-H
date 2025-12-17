"""
U-Net Implementation.

This architecture is adapted from the Pytorch-UNet repository:
https://github.com/milesial/Pytorch-UNet

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation."
Medical image computing and computer-assisted intervention–MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.regression_model.components.unet_utils import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(
            self,
            n_channels_in,
            bilinear,
            out_activation,
            width_multiplier,
            ):
        
        super(UNet, self).__init__()
        self.n_channels = n_channels_in
        self.bilinear = bilinear
        
        base_channels = int(64 * width_multiplier)
        
        self.inc = DoubleConv(n_channels_in, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, (base_channels * 16) // factor)
        self.up1 = Up((base_channels * 16), (base_channels * 8) // factor, bilinear)
        self.up2 = Up((base_channels * 8), (base_channels * 4) // factor, bilinear)
        self.up3 = Up((base_channels * 4), (base_channels * 2) // factor, bilinear)
        self.up4 = Up((base_channels * 2), base_channels, bilinear)
        self.outc = OutConv(base_channels, 1)
        
        self.out_activation = None
        if out_activation is not None:
            if (out_activation == "None") or (out_activation is None) or (out_activation == "null"):
                self.out_activation = None
            else:
                self.out_activation = out_activation

    def forward(self, x, meta_data):
        x1 = self.inc(x)
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

