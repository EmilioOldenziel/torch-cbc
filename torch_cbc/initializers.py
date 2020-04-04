import torch


def reasoning_init(reasoning: torch.Tensor, n_components, n_classes):
    reasoning.data = torch.zeros((2, 1, n_components * n_classes, n_classes))
    reasoning.data[0] = torch.rand(1, 1, n_components * n_classes, n_classes) / 4
    reasoning.data[1] = 1 - reasoning[0]
    return reasoning
