import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from ontological_audio_embeddings.src.models.customLayers.SublayerConnection import SublayerConnection
from ontological_audio_embeddings.src.utils.utils import cloneModule

class EncoderLayer(nn.Module):
    """
    Uses self-attention, and feed forward
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = cloneModule(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)