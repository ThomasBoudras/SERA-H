import torch
import torch.nn.functional as F
import torch.nn as nn

class MaskedHuber(nn.Module):
    def __init__(self, reduction, beta):
        super(MaskedHuber, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target, metadata):
        mask = ~torch.isnan(target) & ~torch.isnan(pred)  # create a mask where target and pred are not NaN
        return F.smooth_l1_loss(pred[mask], target[mask], reduction=self.reduction, beta=self.beta)
