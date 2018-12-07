import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from ontological_audio_embeddings.src.models.customLayers import LayerNorm

class SublayerConnection(nn.Module):
    """
    Residual connection
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        :param x: input signal
        :param sublayer: sublayer to be applied over x
        :return: residual output of sublayer(x) + x
        """
        return x + self.dropout(sublayer(self.norm(x)))