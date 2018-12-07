import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from ontological_audio_embeddings.utils.utils import cloneModule
from ontological_audio_embeddings.src.models.custom_layers.LayerNorm import LayerNorm


class Encoder(nn.Module):
    """
    Encoder - Stack of N sublayers
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = cloneModule(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
