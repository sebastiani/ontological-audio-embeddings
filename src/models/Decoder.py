import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from ontological_audio_embeddings.src.utils.utils import cloneModule
from ontological_audio_embeddings.src.models.customLayers import LayerNorm


class Decoder(nn.Module):
    """
    Decoder base
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = cloneModule(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, x_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, x_mask, target_mask)
        return self.norm(x)