import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity2D(nn.Module):

    def __init__(self, n_replicas=1, eps=10e-10, activation=F.relu,
                 filter_size=(9, 128, 4, 4)):
        super(CosineSimilarity2D, self).__init__()
        self.n_replicas = n_replicas
        self.eps = eps
        self.activation = activation

        self.register_buffer('constant_filter', torch.ones(filter_size))

    def forward(self, x, y):
        """
            Input:
                x: input tensor of shape (minibatch, in_channels, H, W)
                y: input tensor of shape (minibatch, in_channels, H, W)
            Output
                distances tensor of shape (minibatch, n_components, 1, 1)
        """

        # normalize kernel
        normed_y = y / y.pow(2) \
                        .sum(dim=(1, 2, 3), keepdim=True) \
                        .sqrt() \
                        .clamp(min=self.eps)  # to prevent zero division

        # get norm of signals
        x_norm = F.conv2d(x.pow(2), self.constant_filter) \
                  .sqrt() \
                  .clamp(min=self.eps)  # to prevent zero division

        diss = F.conv2d(x, normed_y) / x_norm

        return self.activation(diss)
