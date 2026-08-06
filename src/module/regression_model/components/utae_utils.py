"""Building blocks (convolutions, temporal aggregation) used by the U-TAE architecture."""

from typing import List, Optional

import torch
import torch.nn as nn


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an inputs tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value: Optional[float] = None) -> None:
        """Initialize the block.

        Args:
            pad_value (Optional[float]): Value used to identify padded time steps that
                should be skipped when applying the shared forward pass on 5-D inputs.
        """
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply `self.forward` shared across the temporal dimension.

        If `inputs` is 4-D (B, C, H, W), applies `forward` directly. If it is 5-D
        (B, T, C, H, W), flattens the batch and temporal dimensions, applies `forward`
        to all (batch x time) positions (skipping padded ones if `pad_value` is set),
        and reshapes the result back to (B, T, C', H', W').

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            torch.Tensor: Output tensor, either (B, C', H', W') or (B, T, C', H', W')
                depending on the shape of `inputs`.
        """
        if len(inputs.shape) == 4:
            return self.forward(inputs)
        else:
            b, t, c, h, w = inputs.shape

            if self.pad_value is not None:
                dummy = torch.zeros(inputs.shape, device=inputs.device, dtype=inputs.dtype)
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = inputs.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(self.out_shape, device=inputs.device, requires_grad=False)
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    """Stack of 2D convolutions, each optionally followed by normalization and ReLU."""

    def __init__(
        self,
        nkernels: List[int],
        norm: str = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        """Build the sequence of convolution (+ norm + ReLU) layers.

        Args:
            nkernels (List[int]): Number of channels for each layer, e.g. [c_in, c_mid, c_out]
                defines `len(nkernels) - 1` convolutions.
            norm (str): Normalization type: "batch" (BatchNorm2d), "instance" (InstanceNorm2d),
                "group" (GroupNorm with `n_groups` groups), or any other value for no normalization.
            k (int): Kernel size of the convolutions.
            s (int): Stride of the convolutions.
            p (int): Padding of the convolutions.
            n_groups (int): Number of groups used when `norm` is "group".
            last_relu (bool): If True, apply ReLU after every convolution; if False, ReLU is
                applied after all but the last convolution.
            padding_mode (str): Padding mode passed to `nn.Conv2d`.
        """
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolution stack.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W').
        """
        return self.conv(inputs)


class ConvBlock(TemporallySharedBlock):
    """Temporally-shared `ConvLayer` wrapper, usable on 4-D or 5-D (batch x time) inputs."""

    def __init__(
        self,
        nkernels: List[int],
        pad_value: Optional[float] = None,
        norm: str = "batch",
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        """Build the underlying `ConvLayer`.

        Args:
            nkernels (List[int]): Number of channels for each layer of the convolution stack.
            pad_value (Optional[float]): Value used to identify padded time steps when called
                through `smart_forward` on 5-D inputs.
            norm (str): Normalization type, see `ConvLayer`.
            last_relu (bool): If True, apply ReLU after every convolution.
            padding_mode (str): Padding mode passed to `nn.Conv2d`.
        """
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolution stack.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W').
        """
        return self.conv(inputs)


class DownConvBlock(TemporallySharedBlock):
    """Temporally-shared downsampling block: strided conv followed by a residual conv pair."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        pad_value: Optional[float] = None,
        norm: str = "batch",
        padding_mode: str = "reflect",
    ) -> None:
        """Build the strided downsampling convolution and the residual convolution pair.

        Args:
            d_in (int): Number of input channels.
            d_out (int): Number of output channels.
            k (int): Kernel size of the strided downsampling convolution.
            s (int): Stride of the strided downsampling convolution.
            p (int): Padding of the strided downsampling convolution.
            pad_value (Optional[float]): Value used to identify padded time steps when called
                through `smart_forward` on 5-D inputs.
            norm (str): Normalization type, see `ConvLayer`.
            padding_mode (str): Padding mode passed to `nn.Conv2d`.
        """
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the strided downsampling convolution followed by the residual conv pair.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Downsampled output tensor of shape (B, C_out, H', W').
        """
        out = self.down(inputs)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    """Upsampling block: transposed convolution merged with a skip connection, then a
    residual conv pair.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        norm: str = "batch",
        d_skip: Optional[int] = None,
        padding_mode: str = "reflect",
    ) -> None:
        """Build the transposed convolution, skip-connection projection, and residual conv pair.

        Args:
            d_in (int): Number of input channels.
            d_out (int): Number of output channels.
            k (int): Kernel size of the transposed (upsampling) convolution.
            s (int): Stride of the transposed (upsampling) convolution.
            p (int): Padding of the transposed (upsampling) convolution.
            norm (str): Normalization type, see `ConvLayer`.
            d_skip (Optional[int]): Number of channels of the skip connection tensor. Defaults
                to `d_out` if not specified.
            padding_mode (str): Padding mode passed to `nn.Conv2d`.
        """
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample `inputs`, concatenate with the projected skip connection, and convolve.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, d_in, H, W) to upsample.
            skip (torch.Tensor): Skip connection tensor of shape (B, d_skip, 2H, 2W).

        Returns:
            torch.Tensor: Output tensor of shape (B, d_out, 2H, 2W).
        """
        out = self.up(inputs)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlockMF(nn.Module):
    """Upsampling block with multi-frame skip-connection coupling (difference or concat)."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        coupling_mode: str,
        norm: str = "batch",
        d_skip: Optional[int] = None,
        padding_mode: str = "reflect",
    ) -> None:
        """Build the transposed convolution, skip-connection projection, and residual conv pair.

        Args:
            d_in (int): Number of input channels.
            d_out (int): Number of output channels.
            k (int): Kernel size of the transposed (upsampling) convolution.
            s (int): Stride of the transposed (upsampling) convolution.
            p (int): Padding of the transposed (upsampling) convolution.
            coupling_mode (str): How the two skip-connection frames are combined: "difference"
                (subtract projected skip features) or "concat" (concatenate them).
            norm (str): Normalization type, see `ConvLayer`.
            d_skip (Optional[int]): Number of channels of each skip connection tensor. Defaults
                to `d_out` if not specified.
            padding_mode (str): Padding mode passed to `nn.Conv2d`.
        """
        super(UpConvBlockMF, self).__init__()
        self.coupling_mode = coupling_mode
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        d_skip_out = 2 * d if self.coupling_mode == "concat" else d
        self.conv1 = ConvLayer(
            nkernels=[d_out + d_skip_out, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(
        self, inputs: torch.Tensor, skip_t1: torch.Tensor, skip_t2: torch.Tensor
    ) -> torch.Tensor:
        """Upsample `inputs` and merge it with the two skip-connection frames.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, d_in, H, W) to upsample.
            skip_t1 (torch.Tensor): Skip connection tensor of the first frame, shape
                (B, d_skip, 2H, 2W).
            skip_t2 (torch.Tensor): Skip connection tensor of the second frame, shape
                (B, d_skip, 2H, 2W).

        Returns:
            torch.Tensor: Output tensor of shape (B, d_out, 2H, 2W).
        """
        out = self.up(inputs)
        if self.coupling_mode == "difference":
            skip = self.skip_conv(skip_t2) - self.skip_conv(skip_t1)
            out = torch.cat([out, skip], dim=1)

        elif self.coupling_mode == "concat":
            out_separation = inputs.shape[1] // 2
            out_t1 = out[:, :out_separation, :, :].contiguous()
            out_t2 = out[:, out_separation:, :, :].contiguous()
            out = torch.cat(
                [out_t1, self.skip_conv(skip_t1), out_t2, self.skip_conv(skip_t2)], dim=1
            )

        else:
            Exception("Give a format for coupling mode valid: either “difference” or “concat”.")
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    """Aggregates a sequence of feature maps into a single feature map, using either a
    (masked) temporal mean or the LTAE attention masks as weights.
    """

    def __init__(self, mode: str = "mean") -> None:
        """Initialize the aggregator.

        Args:
            mode (str): Aggregation mode: "att_group", "att_mean", or "mean" (see
                `UTAE`'s `agg_mode` argument for details).
        """
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate the temporal dimension of `x`.

        Args:
            x (torch.Tensor): Feature maps to aggregate, of shape (B, T, C, H, W).
            pad_mask (Optional[torch.Tensor]): Boolean mask of shape (B, T) indicating padded
                time steps to exclude from the aggregation. If None or all-False, no masking
                is applied.
            attn_mask (Optional[torch.Tensor]): Temporal attention masks from the LTAE, of
                shape (n_heads, B, T, h, w), used when `mode` is "att_group" or "att_mean".

        Returns:
            torch.Tensor: Aggregated feature map of shape (B, C, H, W).
        """
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(
                        attn
                    )
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(
                        attn
                    )
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)
