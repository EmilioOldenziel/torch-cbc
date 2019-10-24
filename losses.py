import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.modules.loss._Loss):

    def __init__(self, margin=0.3):
        super(MarginLoss, self).__init__(margin)
        self.margin = margin

    def forward(self, input, target):
        dp = torch.sum(target * input, dim=-1)
        dm = torch.max(input - target, dim=-1).values
        return F.relu(dm - dp + self.margin)