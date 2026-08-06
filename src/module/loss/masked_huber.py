"""Huber (smooth L1) loss that ignores NaN values in the target and prediction."""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedHuber(nn.Module):
    """Huber loss computed only on entries where neither the target nor the prediction is NaN."""

    def __init__(self, reduction: str, beta: float) -> None:
        """Initializes the loss.

        Args:
            reduction: Reduction mode passed to `F.smooth_l1_loss` (e.g. "mean", "sum", "none").
            beta: Threshold at which the loss transitions from L2 to L1, passed to
                `F.smooth_l1_loss`.
        """
        super(MaskedHuber, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Computes the masked Huber loss between predictions and targets.

        Args:
            pred: Predicted tensor, same shape as `target`.
            target: Target tensor, may contain NaN values which are excluded from the loss.
            metadata: Batch metadata (unused, kept for interface consistency).

        Returns:
            The Huber loss computed over the non-NaN entries, reduced according to
            `self.reduction`.
        """
        mask = ~torch.isnan(target) & ~torch.isnan(
            pred
        )  # create a mask where target and pred are not NaN
        return F.smooth_l1_loss(pred[mask], target[mask], reduction=self.reduction, beta=self.beta)
