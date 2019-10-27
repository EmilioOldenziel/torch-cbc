import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning_layer import ReasoningLayer
from detection_probability_functions import CosineSimilarity2D

class CBCModel(nn.Module):
    def __init__(self, backbone, n_classes, n_components, component_shape):
        super(CBCModel, self).__init__()
        
        self.backbone = backbone
        
        self.components = nn.Parameter(torch.zeros((n_components,) + component_shape)) # (n_components, H, W)
        self.similarity = CosineSimilarity2D()
        self.reasoning_layer = ReasoningLayer(n_components=n_components, n_classes=n_classes) #[1, n_components) -> (1, n_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        y = self.backbone(self.components)

        detection = self.similarity(x, y)

        detection = detection.permute(0,2,3,1)
        detection = detection.squeeze(1).squeeze(1) # (batch size, n_components)

        return self.reasoning_layer(detection) 