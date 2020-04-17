import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .layers import ReasoningLayer
from .cosine2d import CosineSimilarity2D
from .initializers import reasoning_init


class FixedCBCModel(nn.Module):
    """
    CBC model with a fixed backbone and pre-computed component features.
    """

    def __init__(self, backbone, n_classes, n_components, component_shape, data, targets, image_transform):
        super(FixedCBCModel, self).__init__()

        self.backbone = backbone

        self.components = nn.Parameter(  # (n_components, H, W)
            torch.rand((n_components*n_classes,) + component_shape)
        )

        self.reasoning_layer = ReasoningLayer(n_components=n_components,
                                        n_classes=n_classes)

        reasoning_init(self.reasoning_layer.reasoning_probabilities, 
                       n_components, 
                       n_classes)

        import numpy as np
        from PIL import Image

        # set components with class 5 samples for each class
        for class_idx in range(10):
            class_target_indices = np.where(np.array(targets) == class_idx)[0][:5]
            class_components = np.take(data, class_target_indices, axis=0)

            for class_component_idx in range(n_components):

                image_idx = (class_idx*n_components)+class_component_idx
                self.components.data[image_idx] = image_transform(Image.fromarray(class_components[class_component_idx]))

            # set reasoning
            tmp = self.reasoning_layer.reasoning_probabilities.data[0, 0, image_idx, class_idx]
            self.reasoning_layer.reasoning_probabilities.data[0, 0, image_idx, class_idx] = self.reasoning_layer.reasoning_probabilities[-1, 0, image_idx, class_idx]
            self.reasoning_layer.reasoning_probabilities.data[-1, 0, image_idx, class_idx] = tmp

        self.register_buffer('components_feature_map', self.backbone(self.components))

        self.similarity = CosineSimilarity2D(filter_size=self.components_feature_map.shape)  # noqa

    def forward(self, x):

        x = self.backbone(x)
        y = self.backbone(self.components)  # TODO: replace by pre-computed feature map

        detection = self.similarity(x,  y)
        # detection: (batch, n_components, 1, 1)

        detection = detection.squeeze(-1).squeeze(-1)
        # detection: (batch size, n_components)

        return self.reasoning_layer(detection)
