import torch
import torch.nn as nn
import torch.nn.functional as F


class binary_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'binary_entropy'
        self.TYPE = 'prediction'
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        if inputs.dtype != torch.float32:
            inputs = inputs.to(torch.float32)
        if targets.dtype != torch.float32:
            targets = targets.to(torch.float32)

        return F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction=self.reduction)


CRITERIA = {'binary_entropy': binary_entropy}