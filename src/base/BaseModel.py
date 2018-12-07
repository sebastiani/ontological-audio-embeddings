import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pickle
import torch.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ontological_audio_embeddings.src.data_loader.AudioSetDataset import AudioSetDataset
# torch.backends.cudnn.enabled = False

from ontological_audio_embeddings.src.models.PatchGAN import Discriminator, GANLoss
from ontological_audio_embeddings.src.models.Transformer import Transformer

class BaseModel(object):
    def __init__(self, params):
        self.Generator = Transformer
        self.Discriminator = Discriminator

        self.cuda = params['cuda']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.momentum = params['momentum']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.lambda_l1 = params['L1_lambda']

        self.data = params['dataset']

    def train(self, params):
        if self.model is None:
            raise ("ERROR: no model has been specified")

        # Data loading is manual for now

        dataset = AudioSetDataset(params['dataset'], params['label_dict'])
        datsetSize = len(dataset)
        indices = list(range(datsetSize))
        vsplit = int(np.floor(params['validation_split'] * datsetSize))
        tsplit = int(np.floor(params['test_split'] * datsetSize))

        np.random.seed(params['seed'])
        np.random.shuffle(indices)

        train_indices, val_indices = indices[vsplit:], indices[:vsplit]
        train_indices, test_indices = train_indices[tsplit:], indices[:tsplit]

        trainSampler = SubsetRandomSampler(train_indices)
        valSampler = SubsetRandomSampler(val_indices)
        testSampler = SubsetRandomSampler(test_indices)

        trainLoader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            sampler=trainSampler,
            num_workers=4
        )

        valLoader = DataLoader(
            dataset,
            batch_size=min(params['batch_size'], len(val_indices)),
            sampler=valSampler,
            num_workers=4
        )

        testSampler = DataLoader(
            dataset,
            batch_size=min(params['batch_size'], len(test_indices)),
            sampler=testSampler,
            num_workers=4
        )


        # Start training loop
        if self.cuda:
            self.Generator = self.Generator.cuda()
            self.Discriminator = self.Discriminator.cuda()

        criterionGAN = GANLoss()
        criterionL1 = torch.nn.L1Loss()
        #loss_fn = nn.CrossEntropyLoss()
        if self.cuda:
            criterionGAN = criterionGAN.cuda()
            #loss_fn = loss_fn.cuda()

        #optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        G_optimizer = optim.Adam(self.Generator.parameters(),
                                 lr=self.lr, betas=(self.beta1, 0.999))
        D_optimizer = optim.Adam(self.Discriminator.parameters(),
                                 lr=self.lr, betas=(self.beta2, 0.999))

        # remember to set volatile=True for validation input and label
        for epoch in range(self.epochs):
            discriminator_train_running_loss = 0.0
            generator_train_running_loss = 0.0

            for i, (samples, noisy_samples, labels) in enumerate(trainLoader):
                inputs, noisy_inputs, labels = torch.from_numpy(samples), torch.from_numpy(noisy_samples), torch.from_numpy(labels).long()

                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                # Forward-step: generate fake samples
                fake_samples = self.Generator(inputs.float())

                # Calculate Discriminator Loss
                # There's some fake_B detaching???? check that out!
                self.set_requires_grad(self.Discriminator, True)
                D_optimizer.zero_grad()

                # pairs up generated samples with noisy real samples
                fake_pairs = torch.cat((noisy_inputs, fake_samples), 1)
                pred_fake = self.Discriminator(fake_pairs)
                D_loss_fake = criterionGAN(pred_fake, True)

                # pairs up real samples and noisy samples
                true_pairs = torch.cat((noisy_inputs, inputs), 1)
                pred_real = self.Discriminator(true_pairs)
                D_loss_real = criterionGAN(pred_real)

                discriminatorLoss = 0.5*(D_loss_fake + D_loss_real)

                discriminatorLoss.backward()

                D_optimizer.step()
                discriminator_train_running_loss == discriminatorLoss.cpu().item()

                # Calculate Generator Loss
                self.set_requires_grad(self.Discriminator, False)
                G_optimizer.zero_grad()

                pred_fake = self.Discriminator(fake_pairs)
                G_loss = criterionGAN(pred_fake)
                G_loss_L1 = criterionL1(fake_samples, inputs) * self.lambda_l1

                generatorLoss = G_loss + G_loss_L1
                generatorLoss.backward()
                G_optimizer.step()

                generator_train_running_loss += generatorLoss.cpu().item()

                if i % 5 == 0:
                    print('[%d, %5d] train loss: %.3f' %
                          (epoch + 1, i + 1, train_running_loss / 100))
                    train_running_loss = 0.0

                # ADD CHECKPOINTING HERE

        print("Finished training!")
        print("Saving model to %s" % (params['saved_models'] + 'model_weights.pt'))
        torch.save(self.model.state_dict(), params['saved_models'] + 'model_weights.pt')

    def predict(self, params):
        self.model.load_state_dict(torch.load(params['saved_models'] + 'model_weights.pt'))
        self.model.eval()
        raise ("Not implemented yet")

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
