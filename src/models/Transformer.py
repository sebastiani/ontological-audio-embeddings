import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time
from copy import deepcopy
from torch.autograd import Variable

from ontological_audio_embeddings.src.models.customLayers.MultiHeadAttention import MultiHeadedAttention
from ontological_audio_embeddings.src.models.customLayers.PositionWiseFF import PositionwiseFeedForward
from ontological_audio_embeddings.src.models.customLayers.PositionalEncoding import PositionalEncoding
from ontological_audio_embeddings.src.models.customLayers.EncoderLayer import EncoderLayer
from ontological_audio_embeddings.src.models.customLayers.DecoderLayer import DecoderLayer
from ontological_audio_embeddings.src.models.Decoder import Decoder
from ontological_audio_embeddings.src.models.Encoder import Encoder
from ontological_audio_embeddings.src.models.EncoderDecoder import EncoderDecoder
from ontological_audio_embeddings.src.models.Generator import Generator


class Transformer(nn.Module):

    def __init__(self, target_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        #self.x_vocab = x_vocab
        self.target_vocab = target_vocab
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout

        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(self.d_model, deepcopy(attn), deepcopy(ff), self.dropout), N),
            Decoder(DecoderLayer(self.d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), self.dropout), self.N),
            deepcopy(position),
            deepcopy(position),
            Generator(self.d_model, None) #for now

        )

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, x, target, x_mask, target_mask):
        return self.model.forward(x, target, x_mask, target_mask)