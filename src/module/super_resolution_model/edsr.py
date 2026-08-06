"""
EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) implementation.

This architecture is adapted from the official PyTorch implementation of the original paper:
https://github.com/sanghyun-son/EDSR-PyTorch

Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution."
Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src import global_utils as utils
from src.module.super_resolution_model.components.edsr_utils import ResBlock, Upsampler, conv2d

log = utils.get_logger(__name__)


class EDSR(nn.Module):
    """Enhanced Deep Residual Network for single image super-resolution.

    Composed of a head (initial feature extraction), a body (stack of residual
    blocks) and a tail (upsampling + reconstruction), following the original
    EDSR architecture. Supports both single-composite inputs and time series
    of inputs, and can optionally be initialized from a pretrained checkpoint.
    """

    def __init__(
        self,
        n_resblocks: int,
        scale: int,
        n_feats: int,
        n_channels: int,
        res_scale: float,
        pretrained_model_path: Optional[str],
        input_type: str,
        freeze_model: bool,
    ) -> None:
        """Initializes the EDSR model.

        Args:
            n_resblocks: Number of residual blocks in the body of the network.
            scale: Super-resolution upscaling factor (e.g. 2, 3, 4).
            n_feats: Number of feature channels used in the body of the network.
            n_channels: Number of input/output channels of the model.
            res_scale: Scaling factor applied to the output of each residual block.
            pretrained_model_path: Path to a pretrained EDSR checkpoint to partially
                initialize the model from, or None to skip pretrained initialization.
            input_type: Either "TIMESERIES" or any other value (e.g. "COMPOSITES") to
                select the forward pass variant used at inference time.
            freeze_model: If True, freezes all parameters of the model after
                initialization (and after loading pretrained weights, if any).
        """
        super(EDSR, self).__init__()
        self.forward_method = (
            self.forward_timeseries if input_type == "TIMESERIES" else self.forward_composites
        )

        kernel_size = 3
        act = nn.ReLU(True)

        self.n_channels = n_channels
        self.scale = scale
        self.pretrained_model_path = (
            Path(pretrained_model_path).resolve() if pretrained_model_path is not None else None
        )
        self.input_type = input_type

        # define head module
        m_head = [conv2d(n_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv2d, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv2d(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv2d, scale, n_feats, act=False),
            conv2d(n_feats, n_channels, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if self.pretrained_model_path is not None:
            self.load_partial_weight()

        if freeze_model:
            self.freeze_model()

    def freeze_model(self) -> None:
        """Freezes all parameters of the head, body and tail modules (sets `requires_grad` to False)."""
        for param in self.head.parameters():
            param.requires_grad = False
        for param in self.body.parameters():
            param.requires_grad = False
        for param in self.tail.parameters():
            param.requires_grad = False

    def load_partial_weight(self) -> None:
        """Loads a pretrained RGB EDSR checkpoint into this (possibly multi-channel) model.

        The pretrained checkpoint is trained on 3-channel (RGB) images, while this model
        may use a different number of input/output channels (`self.n_channels`). The head
        and tail weights/biases are adapted by averaging across the pretrained channels and
        expanding to `self.n_channels`, with the first 3 channels re-initialized from the
        pretrained RGB weights (reordered to BGR to match Sentinel-2 band order). The tail
        weights/biases are also cropped to match the model's `scale`. The body weights are
        loaded unchanged. Weights are loaded with `strict=False` since the head/tail shapes
        may still differ slightly from the checkpoint.
        """
        log.info(
            f"Using the pre-trained model {self.pretrained_model_path.name} to initialise the model"
        )
        load_from = torch.load(
            self.pretrained_model_path, map_location=torch.device("cpu"), weights_only=True
        )

        # We only change weight of the head and tail of the model, the body does not need to be changed

        # Update "head.0.weight"
        module_tensor = load_from["head.0.weight"]
        mean_weight = module_tensor.mean(dim=1, keepdim=True)
        expanded_weight = mean_weight.expand(-1, self.n_channels, -1, -1).clone()
        expanded_weight[:, :3, :, :] = module_tensor[
            :, [2, 1, 0], :, :
        ]  # Sentinel-2 start with BGR
        load_from["head.0.weight"] = expanded_weight

        # Update "tail.1.weight"
        module_tensor = load_from["tail.1.weight"]
        mean_weight = module_tensor.mean(dim=0, keepdim=True)
        expanded_weight = mean_weight.expand(self.n_channels, -1, -1, -1).clone()
        expanded_weight[:3, :, :, :] = module_tensor[
            [2, 1, 0], :, :, :
        ]  # Sentinel-2 start with BGR
        load_from["tail.1.weight"] = expanded_weight

        # Update "tail.1.bias"
        module_tensor = load_from["tail.1.bias"]
        mean_bias = module_tensor.mean().unsqueeze(0)
        expanded_bias = mean_bias.expand(self.n_channels).clone()
        expanded_bias[:3] = module_tensor[[2, 1, 0]]  # Sentinel-2 start with BGR
        load_from["tail.1.bias"] = expanded_bias

        load_from["tail.0.0.weight"] = load_from["tail.0.0.weight"][: 256 * self.scale, :, :, :]
        load_from["tail.0.0.bias"] = load_from["tail.0.0.bias"][: 256 * self.scale]
        load_from["tail.0.2.weight"] = load_from["tail.0.2.weight"][: 256 * self.scale, :, :, :]
        load_from["tail.0.2.bias"] = load_from["tail.0.2.bias"][: 256 * self.scale]

        self.load_state_dict(load_from, strict=False)

    def forward_timeseries(
        self, inputs: torch.Tensor, targets: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Runs the forward pass on a time series of inputs.

        Args:
            inputs: Input tensor of shape (B, T, C, H, W), where T is the number of
                time steps in the series.
            targets: Target tensor (unused, kept for interface consistency).
            metadata: Batch metadata (unused, kept for interface consistency).

        Returns:
            Output tensor of shape (B, T, n_channels, H * scale, W * scale).
        """
        # inputs: (B, T, C, H, W)
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        inputs = self.head(inputs)
        res = self.body(inputs)
        res += inputs
        inputs = self.tail(res)
        inputs = inputs.view(B, T, self.n_channels, inputs.shape[-2], inputs.shape[-1])
        return inputs

    def forward_composites(
        self, inputs: torch.Tensor, targets: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Runs the forward pass on a single (non-temporal) composite input.

        Args:
            inputs: Input tensor of shape (B, C, H, W).
            targets: Target tensor (unused, kept for interface consistency).
            metadata: Batch metadata (unused, kept for interface consistency).

        Returns:
            Output tensor of shape (B, n_channels, H * scale, W * scale).
        """
        # inputs: (B, C, H, W)
        inputs = self.head(inputs)
        res = self.body(inputs)
        res += inputs
        inputs = self.tail(res)
        return inputs

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Runs the forward pass selected at init time (`forward_timeseries` or `forward_composites`).

        Args:
            inputs: Input tensor, either (B, T, C, H, W) or (B, C, H, W) depending on `input_type`.
            targets: Target tensor (unused, kept for interface consistency).
            metadata: Batch metadata (unused, kept for interface consistency).

        Returns:
            Super-resolved output tensor, with the same layout as `inputs` but with
            spatial dimensions scaled by `self.scale` and `self.n_channels` channels.
        """
        return self.forward_method(inputs, targets, metadata)
