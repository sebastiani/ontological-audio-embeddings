import torch
import torch.nn as nn
from ontological_audio_embeddings.src.models.Transformer import Transformer
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

class Generator(nn.Module):
    def __init__(self, input_dim, num_filters=16, norm_layer=nn.BatchNorm1d, use_sigmoid=False):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        kernel_size = 160   # corresponds to 10ms length kernel
        pad_size = 1
        nf_mult = 2

        stride = 2

        self.encoder_layers = []
        previous_mult = input_dim

        # Encoder
        self.encoder1 = nn.Conv1d(previous_mult, num_filters, kernel_size, stride=stride)
        self.encoder2 = nn.Conv1d(num_filters, num_filters, kernel_size, stride=stride)
        self.encoder3 = norm_layer(num_filters)

        self.encoder4 = nn.Conv1d(num_filters, num_filters*2, kernel_size, stride=stride)
        self.encoder5 = nn.Conv1d(num_filters*2, num_filters * 2, kernel_size, stride=stride)
        self.encoder6 = norm_layer(num_filters*2)

        self.encoder7 = nn.Conv1d(num_filters*2, num_filters * 3, kernel_size, stride=stride)
        self.encoder8 = nn.Conv1d(num_filters*3, num_filters * 3, kernel_size, stride=stride)
        self.encoder9 = norm_layer(num_filters*3)

        self.encoder10 = nn.Conv1d(num_filters*3, num_filters * 4, kernel_size, stride=stride)
        self.encoder11 = nn.Conv1d(num_filters*4, num_filters * 4, kernel_size, stride=stride)
        self.encoder12 = norm_layer(num_filters * 4)

        # Decoder
        self.decoder1 = nn.ConvTranspose1d(num_filters*4, num_filters * 3, kernel_size+1, stride=stride)
        self.decoder2 = nn.ConvTranspose1d(num_filters * 3, num_filters * 3, kernel_size + 1, stride=stride)
        self.decoder3 = norm_layer(num_filters * 3)

        self.decoder4 = nn.ConvTranspose1d(num_filters * 3, num_filters * 2, kernel_size + 1, stride=stride)
        self.decoder5 = nn.ConvTranspose1d(num_filters * 2, num_filters * 2, kernel_size + 1, stride=stride)
        self.decoder6 = norm_layer(num_filters * 2)

        self.decoder7 = nn.ConvTranspose1d(num_filters * 2, num_filters, kernel_size-2, stride=stride)
        self.decoder8 = nn.ConvTranspose1d(num_filters, num_filters, kernel_size + 1, stride=stride)
        self.decoder9 = norm_layer(num_filters)

        self.decoder10 = nn.ConvTranspose1d(num_filters, num_filters, kernel_size-1, stride=stride)


        self.encoder_relu = nn.LeakyReLU(0.2, True)
        self.decoder_relu = nn.LeakyReLU(0.2, True)


    def forward(self, x):
        x = self.encoder_relu(self.encoder1(x))
        x = self.encoder2(x)
        x1 = self.encoder_relu(self.encoder3(x))

        x = self.encoder_relu(self.encoder4(x1))
        x = self.encoder5(x)
        x2 = self.encoder_relu(self.encoder6(x))

        x = self.encoder_relu(self.encoder7(x2))
        x = self.encoder8(x)
        x3 = self.encoder_relu(self.encoder9(x))

        x = self.encoder_relu(self.encoder10(x3))
        x = self.encoder12(x)

        # Decoding
        x = self.decoder_relu(self.decoder1(x)) + x3
        x = self.decoder2(x)
        x = self.decoder_relu(self.decoder3(x))


        x = self.decoder_relu(self.decoder4(x)) + x2
        x = self.decoder5(x)
        x = self.decoder_relu(self.decoder6(x))

        x = self.decoder_relu(self.decoder7(x)) + x1
        x = self.decoder8(x)
        x = self.decoder_relu(self.decoder9(x))

        x = self.decoder_relu(self.decoder10(x))
        print('Decoder ', x.size())

        return x

    def getEmbedding(self, x):
        x = self.encoder_relu(self.encoder1(x))
        x = self.encoder2(x)
        x = self.encoder_relu(self.encoder3(x))

        x = self.encoder_relu(self.encoder4(x))
        x = self.encoder5(x)
        x = self.encoder_relu(self.encoder6(x))

        x = self.encoder_relu(self.encoder7(x))
        x = self.encoder8(x)
        x = self.encoder_relu(self.encoder9(x))

        x = self.encoder_relu(self.encoder10(x))
        x = self.encoder12(x)

        return(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filters=32, n_layers=4, norm_layer=nn.BatchNorm1d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kernel_size = 160
        pad_size = 0
        stride = 3
        sequence = [
            nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv1d(num_filters*nf_mult_prev, num_filters*nf_mult,
                          kernel_size=kernel_size, stride=stride, padding=pad_size, bias=False),
                norm_layer(num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv1d(num_filters * nf_mult_prev, num_filters*nf_mult,
                      kernel_size=kernel_size, stride=stride, padding=pad_size, bias=False),
            norm_layer(num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence +=  [nn.Conv1d(num_filters * nf_mult, 1, kernel_size=kernel_size, stride=stride, padding=pad_size)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

