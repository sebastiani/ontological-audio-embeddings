import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if use_lsgan:
            self.loss = nn.MSELoss()

        else:
            self.loss = nn.BCELoss()

        def get_target_tensor(self, input, target_is_real):
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            return target_tensor.expand_as(input)

        def __call__(self, input, target_is_real):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_fake_label)



class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filters=64, n_layers=3, norm_layer=nn.BatchNorm1d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kernel_size = 4
        pad_size = 1
        sequence = [
            nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv1d(num_filters*nf_mult_prev, num_filters*nf_mult,
                          kernel_size=kernel_size, stride=2, padding=pad_size, bias=False),
                norm_layer(num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv1d(num_filters * nf_mult_prev, num_filters*nf_mult,
                      kernel_size=kernel_size, stride=1, padding=pad_size, bias=False),
            norm_layer(num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence +=  [nn.Conv1d(num_filters * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=pad_size)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

