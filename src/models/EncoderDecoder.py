import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    """
    Base encoder-decoder Transformer architecture
    """
    def __init__(self, encoder, decoder, input_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed #maybe unnecessaru
        self.target_embed = target_embed #maybe unnecessary
        self.generator = generator

    def forward(self, x, target, x_mask, target_mask):
        return self.decode(self.encode(x, x_mask), x_mask, target, target_mask)

    def encode(self, x, x_mask):
        return self.encoder(self.input_embed(x), x_mask)

    def decode(self, memory, x_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, x_mask, target_mask)