import torch
import torch.nn.functional as F
import torch.nn as nn

class MaskedHuber(nn.Module):
    def __init__(self, reduction, beta):
        super(MaskedHuber, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target):
        mask = ~torch.isnan(target)  # create a mask where target is not NaN
        return F.smooth_l1_loss(pred[mask], target[mask], reduction=self.reduction, beta=self.beta)
