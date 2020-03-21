import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .reasoning_layer import ReasoningLayer
from .cosine2d import CosineSimilarity2D


class CBCModel(nn.Module):
    def __init__(self, backbone, n_classes, n_components, component_shape):
        super(CBCModel, self).__init__()

        self.backbone = backbone

        self.components = nn.Parameter(  # (n_components, H, W)
            torch.rand((n_components,) + component_shape).uniform_(0.45, 0.55)
        )

        self.similarity = CosineSimilarity2D()
        self.reasoning_layer = ReasoningLayer(n_components=n_components,
                                              n_classes=n_classes)

    def forward(self, x):

        x = self.backbone(x)
        y = copy.deepcopy(self.backbone)(self.components)

        detection = self.similarity(x, y)
        # detection: (batch, n_components, 1, 1)

        detection = detection.squeeze(-1).squeeze(-1)
        # detection: (batch size, n_components)

        return self.reasoning_layer(detection)
