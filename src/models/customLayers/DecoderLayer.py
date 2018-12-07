import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from ontological_audio_embeddings.src.utils.utils import cloneModule
from ontological_audio_embeddings.src.models.customLayers import LayerNorm
from ontological_audio_embeddings.src.models.customLayers.SublayerConnection import SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, x_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = x_attn
        self.feed_forward = feed_forward
        self.sublayer = cloneModule(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, x_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, x_mask))
        return self.sublayer[2](x, self.feed_forward)

