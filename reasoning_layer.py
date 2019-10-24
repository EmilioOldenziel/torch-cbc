import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningLayer(nn.Module):
    def __init__(self, n_classes, input_shape, n_replicas=1, eps=1e-8):
        super(ReasoningLayer, self).__init__()
        self.n_classes = n_classes
        self.n_replicas = n_replicas
        self.eps = eps

        self.reasoning_probabilities = nn.Parameter(
            torch.FloatTensor(2, self.n_replicas, input_shape, self.n_classes).uniform_(0, 1)
        )

    def forward(self, x):
        positive_kernel = self.reasoning_probabilities[0]
        negative_kernel = (1-positive_kernel) * self.reasoning_probabilities[1]

        probs = x.matmul((positive_kernel - negative_kernel) \
                + torch.sum(negative_kernel, 1)) \
                / (torch.sum(positive_kernel + negative_kernel, 1) + self.eps)

        # squeeze replica dimension if one.
        if self.n_replicas == 1:
            probs = probs.squeeze(0)
        else:
            pass
            # TODO: permute probs

        return probs