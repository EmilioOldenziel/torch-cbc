import torch
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


class ReasoningLayer(nn.Module):
    def __init__(self, n_components, n_classes, n_replicas=1, eps=1e-08, initialization='random'):
        super(ReasoningLayer, self).__init__()
        self.n_classes = n_classes
        self.n_replicas = n_replicas
        self.eps = eps

        if initialization == 'random':
            probabilities_init = torch.rand(2, self.n_replicas, n_components, self.n_classes)
        if initialization == 'zeros':
            probabilities_init = torch.zeros(2, self.n_replicas, n_components, self.n_classes)

        self.reasoning_probabilities = nn.Parameter(
            probabilities_init
        )

    def forward(self, x):
        positive_kernel = self.reasoning_probabilities[0].clamp(0, 1)
        negative_kernel = (1-positive_kernel) * self.reasoning_probabilities[1].clamp(0, 1)  # noqa

        probs = (torch.matmul(x, (positive_kernel - negative_kernel))
            + torch.sum(negative_kernel, 1)) \
            / torch.sum(positive_kernel + negative_kernel, 1).clamp(min=self.eps)  # noqa

        # squeeze replica dimension if it is 1.
        if self.n_replicas == 1:
            probs = probs.squeeze(0)
        else:
            raise NotImplementedError()
            # TODO: permute probs

        return probs
