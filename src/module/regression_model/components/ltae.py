"""Lightweight Temporal Attention Encoder (L-TAE) and its building blocks, used by U-TAE."""

import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder applied independently to every pixel position
    of an image time series, mapping the sequence of embeddings to a single feature map.
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp: List[int] = [256, 128],
        dropout: float = 0.2,
        d_model: Optional[int] = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ) -> None:
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.

        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated
                outputs of the attention heads.
            dropout (float): Dropout rate.
            d_model (Optional[int]): If specified, the input tensors will first be processed
                by a fully connected layer to project them into a feature space of dimension
                `d_model`.
            T (int): Period to use for the positional encoding.
            return_att (bool): If True, the module returns the attention masks along with the
                embeddings (default False).
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        return_comp: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode an image time series into a single feature map via temporal attention.

        Args:
            x (torch.Tensor): Input embeddings of shape (B, T, C, H, W).
            batch_positions (Optional[torch.Tensor]): Temporal positions of shape (B, T), used
                for the positional encoding. Required if `positional_encoding` was enabled.
            pad_mask (Optional[torch.Tensor]): Boolean mask of shape (B, T) indicating padded
                time steps to exclude from the attention computation.
            return_comp (bool): If True, also propagate the raw attention compatibility scores
                (passed through to `MultiHeadAttention`).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output feature map of
                shape (B, mlp[-1], H, W). If `self.return_att` is True, also returns the
                attention masks of shape (n_head, B, T, H, W).
        """
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len).to(out.device)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        """Initialize the shared learnable query and the key projection.

        Args:
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors (per head).
            d_in (int): Dimension of the input value vectors.
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(
        self, v: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, return_comp: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute multi-head attention using a shared learnable query.

        Args:
            v (torch.Tensor): Input embeddings of shape (B, T, d_in), used both to derive the
                keys and as the values.
            pad_mask (Optional[torch.Tensor]): Boolean mask of shape (B, T) indicating padded
                time steps to exclude from the attention computation.
            return_comp (bool): If True, also return the raw (pre-softmax) attention scores.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                The output embeddings of shape (n_head, B, d_in // n_head) and the attention
                weights of shape (n_head, B, T); additionally the raw attention scores if
                `return_comp` is True.
        """
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        """Initialize the attention temperature and dropout.

        Args:
            temperature (float): Scaling factor applied to the attention logits (typically
                `sqrt(d_k)`).
            attn_dropout (float): Dropout rate applied to the attention weights.
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_comp: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape ((n_head*B), d_k).
            k (torch.Tensor): Key tensor of shape ((n_head*B), T, d_k).
            v (torch.Tensor): Value tensor of shape ((n_head*B), T, d_in // n_head).
            pad_mask (Optional[torch.Tensor]): Boolean mask of shape ((n_head*B), T) indicating
                padded time steps, masked out with a large negative value before the softmax.
            return_comp (bool): If True, also return the raw (pre-softmax) attention scores.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                The attention output and the attention weights; additionally the raw attention
                scores if `return_comp` is True.
        """
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for temporal positions, as in the Transformer."""

    def __init__(
        self, d: int, T: int = 1000, repeat: Optional[int] = None, offset: int = 0
    ) -> None:
        """Precompute the sinusoid frequency denominators.

        Args:
            d (int): Dimension of the positional encoding.
            T (int): Period used to compute the sinusoid frequencies.
            repeat (Optional[int]): If specified, the encoding is repeated `repeat` times
                along the last dimension (used to give each attention head its own copy).
            offset (int): Starting index used when computing the sinusoid frequencies.
        """
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        """Compute the sinusoidal positional encoding for a batch of temporal positions.

        Args:
            batch_positions (torch.Tensor): Temporal positions of shape (B, T).

        Returns:
            torch.Tensor: Positional encoding of shape (B, T, C), where C is `d` (or `d * repeat`
                if `repeat` is set).
        """
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)

        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)

        return sinusoid_table
