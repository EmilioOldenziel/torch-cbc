import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.modules.loss._Loss):

    def __init__(self, margin=0.3, size_average=None, reduce=None,
                 reduction='mean'):
        super(MarginLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input_, target):
        dp = torch.sum(target * input_, dim=-1)
        dm = torch.max(input_ - target, dim=-1).values
        return F.relu(dm - dp + self.margin)
