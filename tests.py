import torch
import numpy as np
from detection_probability_functions import CosineSimilarity2D
from losses import MarginLoss


def test_cosine_simularity_2D(a, b):

    cos = CosineSimilarity2D()

    return cos(x, y)


def test_margin_loss(a, b):
    lossfunction = MarginLoss()
    return lossfunction(a, b)

# batch, channels, W, H
inputs = [
    [np.ones((1,24,24,3)),          np.ones((1,24,24,3))],
    [np.zeros((1,24,24,3)),         np.zeros((1,24,24,3))],
    [np.random.randn(1,24,24,3),    np.random.randn(1,24,24,3)]
]

for a, b in inputs:
    x = torch.as_tensor(a, dtype=torch.float)
    y = torch.as_tensor(b, dtype=torch.float)
    print(test_cosine_simularity_2D(x, y).shape)
    #print(test_margin_loss(x,y))