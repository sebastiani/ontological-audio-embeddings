import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


class Generator(nn.Module):
    """
    Output layer, will generate the audio samples using linear output
    """
    def __init__(self, model_dim, output_dim):
        super(Generator, self).__init__()
        self.proj = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # Using relu because magnitudes>0, will have to use some linear scaling later maybe
        return nn.ReLU(self.proj(x))