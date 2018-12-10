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
        for i in range(0, 4):
            mult = nf_mult**i
            self.encoder_layers.append(nn.Conv1d(previous_mult, num_filters*mult, kernel_size, stride=stride))
            self.encoder_layers.append(nn.Conv1d(num_filters*mult, num_filters*mult, kernel_size, stride=stride))
            self.encoder_layers.append(norm_layer(num_filters*mult))
            previous_mult = num_filters * mult
        self.encoder_relu = nn.LeakyReLU(0.2, True)

        self.decoder_layers = []


        for i in reversed(range(0, 3)):
            mult = nf_mult**i
            self.decoder_layers.append(nn.ConvTranspose1d(previous_mult, num_filters * mult, kernel_size+1, stride=stride))
            self.decoder_layers.append(nn.ConvTranspose1d(num_filters * mult, num_filters * mult, kernel_size+1, stride=stride))
            self.decoder_layers.append(norm_layer(num_filters * mult))
            previous_mult = num_filters * mult

        self.decoder_layers.append(nn.ConvTranspose1d(previous_mult, 1, kernel_size, stride=stride))
        self.decoder_relu = nn.LeakyReLU(0.2, True)


    def forward(self, x):
        encoder_outputs = []
        output = x
        count = 0
        print("input ", x.size)
        print("Num of layers: ", len(self.encoder_layers))
        for i, layer in enumerate(self.encoder_layers):
            if count <= 2:
                output = self.encoder_relu(layer(output))
                #print("Encoder Layer", i, ": ", output.size())
                count += 1
            else:
                output = layer(output)
                if len(encoder_outputs) <= 3:
                    print("appending ", output.size())
                    encoder_outputs.append(output)
                count = 0

        print("Num of highways ", len(encoder_outputs))
        count = 0
        i = 2
        for layer in self.decoder_layers:
            output = self.decoder_relu(layer(output))
            print("Decoder layer ", output.size())
            if count == 2:
                print("Adding ", output.size())
                print("with ", encoder_outputs[i].size())
                output = output + encoder_outputs[i]
                count = 0
                i -= 1

            count += 1

        output = self.decoder_relu(self.decoder_layers[-1](output))
        print("Output size ", output.size())
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filters=64, n_layers=3, norm_layer=nn.BatchNorm1d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kernel_size = 160
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

