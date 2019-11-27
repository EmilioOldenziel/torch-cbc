import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarity2D(nn.Module):

    def __init__(self,n_replicas=1):
        super(CosineSimilarity2D, self).__init__()
        self.n_replicas = n_replicas
        self.rank = 2
        self.eps = 10e-10
        self.activation = F.relu

    """
    x: input tensor of shape (minibatch, in_channels, H, W)
    y: input tensor of shape (minibatch, in_channels, H, W)
    """

    def forward(self, x, y):

        # normalize kernel
        normed_y = y /  y.pow(2).sum(dim=(1, 2, 3), keepdim=True).sqrt().clamp(min=self.eps)

        # get norm of signals
        x_norm = F.conv2d(x.pow(2), torch.ones(y.shape)).sqrt().clamp(min=self.eps)

        diss = F.conv2d(x, normed_y) / x_norm

        return self.activation(diss)