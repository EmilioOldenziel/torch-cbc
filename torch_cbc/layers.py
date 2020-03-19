import torch.nn as nn
import torch.nn.functional as F

from torch_cbc.constraints import EuclideanNormalization


class ConstrainedConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(ConstrainedConv2d, self).__init__(*args, **kwargs)
        self.enc = EuclideanNormalization()

    def forward(self, input):
        return F.conv2d(input, self.enc(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
