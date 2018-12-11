import os
import torch
import datetime
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ontological_audio_embeddings.src.data_loader.AudioSetDataset import AudioSetDataset
from tensorboardX import SummaryWriter

from ontological_audio_embeddings.src.models.PatchGAN import Discriminator, Generator, GANLoss


#from ontological_audio_embeddings.src.models.Transformer import Transformer

class BaseModel(object):
    def __init__(self, params):
        #self.Generator = Transformer
        self.Generator = Generator(input_dim=160083, p=params['dropout'])
        self.Discriminator = Discriminator(input_dim=160083)

        self.cuda = params['cuda']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.lr1 = params['learning_rate']
        self.lr2 = params['learning_rate']
        self.momentum = params['momentum']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.lambda_l1 = params['L1_lambda']

        self.data = params['dataset']
        self.model_name = params['model_name']

        self.epoch_count = params['epoch_count']
        self.niter = params['niter']
        self.niter_decay = params['niter_decay']

        self.writer = SummaryWriter()

    def train(self, params):
        if self.Generator is None and self.Discriminator is None:
            raise ("ERROR: no model has been specified")

        # Initializing model checkpoint directory
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
        chkptDir =  os.path.join(params['saved_models'], params['model_name'], timestamp)
        if not os.path.isdir(chkptDir):
            os.makedirs(chkptDir)

        embeddingDir =  os.path.join(params['saved_embeddings'], timestamp)
        if not os.path.isdir(embeddingDir):
            os.makedirs(embeddingDir)

        dataset = AudioSetDataset(params['dataset'], params['label_dict'])
        datsetSize = len(dataset)
        indices = list(range(datsetSize))
        tsplit = int(np.floor(params['test_split'] * datsetSize))

        train_indices, test_indices = indices[tsplit:], indices[:tsplit]

        np.random.seed(params['seed'])
        np.random.shuffle(indices)

        trainSampler = SubsetRandomSampler(train_indices)
        testSampler = SubsetRandomSampler(test_indices)

        trainLoader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            sampler=trainSampler,
            num_workers=4
        )

        testLoader = DataLoader(
            dataset,
            batch_size=min(params['batch_size'], len(test_indices)),
            sampler = testSampler,
            num_workers=4
        )

        print('Loading models')
        # Start training loop
        if self.cuda:
            self.Generator = self.Generator.cuda()
            self.Discriminator = self.Discriminator.cuda()

        criterionGAN = GANLoss()
        criterionL1 = torch.nn.L1Loss()

        print("Loading criterions")
        if self.cuda:
            criterionGAN = criterionGAN.cuda()
            criterionL1 = criterionL1.cuda()


        G_optimizer = optim.Adam(self.Generator.parameters(),
                                 lr=self.lr, betas=(self.beta1, 0.999))
        D_optimizer = optim.Adam(self.Discriminator.parameters(),
                                 lr=self.lr, betas=(self.beta2, 0.999))

        scheduler1 = self.scheduler(G_optimizer)
        scheduler2 = self.scheduler(D_optimizer)

        for epoch in range(self.epochs):
            print('Epoch ', epoch)
            discriminator_train_running_loss = 0.0
            generator_train_running_loss = 0.0

            for i, (samples, noisy_samples, _) in enumerate(trainLoader):
                print("Processing batch ", i)
                inputs, noisy_inputs = torch.from_numpy(samples), torch.from_numpy(noisy_samples)

                if self.cuda:
                    inputs = inputs.cuda()
                    noisy_inputs = noisy_inputs.cuda()
                inputs = Variable(inputs)
                noisy_inputs = Variable(noisy_inputs)

                # Forward-step: generate fake samples
                fake_samples = self.Generator(noisy_inputs.float())

                # Calculate Discriminator Loss
                self.set_requires_grad(self.Discriminator, True)
                D_optimizer.zero_grad()

                # pairs up generated samples with noisy real samples
                fake_pairs = torch.cat((noisy_inputs, fake_samples), 1)
                pred_fake = self.Discriminator(fake_pairs.detach())  # detaching so backprop doesnt go to the Generator
                D_loss_fake = criterionGAN(pred_fake, False)

                # pairs up real samples and noisy samples
                true_pairs = torch.cat((noisy_inputs, inputs), 1)
                pred_real = self.Discriminator(true_pairs)
                D_loss_real = criterionGAN(pred_real, True)

                discriminatorLoss = 0.5*(D_loss_fake + D_loss_real)
                self.writer.add_scalar('discriminator/loss', discriminatorLoss, i)

                discriminatorLoss.backward()

                D_optimizer.step()
                discriminator_train_running_loss == discriminatorLoss.cpu().item()

                # Calculate Generator Loss
                self.set_requires_grad(self.Discriminator, False)
                G_optimizer.zero_grad()

                pred_fake = self.Discriminator(fake_pairs)
                G_loss = criterionGAN(pred_fake)  # check that it isnt detached
                G_loss_L1 = criterionL1(fake_samples, inputs) * self.lambda_l1

                generatorLoss = G_loss + G_loss_L1
                self.writer('generator/loss', generatorLoss, i)
                generatorLoss.backward()
                G_optimizer.step()

                generator_train_running_loss += generatorLoss.cpu().item()

                if i % 5 == 0:
                    print('[%d, %5d] train loss: %.3f' %
                          (epoch + 1, i + 1, train_running_loss / 100))
                    train_running_loss = 0.0
                #ERASE THIS
                break


            print("Saving checkpoints...")
            gfname = 'gen_epoch_{epoch}_weights.pt'.format(epoch=epoch)
            chkptName = os.path.join(chkptDir, gfname)
            torch.save(self.Generator.state_dict(), chkptName)

            # Update learning rate
            scheduler1.step()
            scheduler2.step()
            lr1 = G_optimizer.param_groups[0]['lr']
            lr2 = D_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr1)
            print('learning rate = %.7f' % lr2)

            dfname = 'dis_epoch_{epoch}_weights.pt'.format(epoch=epoch)
            chkptName = os.path.join(chkptDir, dfname)
            torch.save(self.Discriminator.state_dict(), chkptName)

            if epoch % 2 == 0:
                self.set_requires_grad(self.Generator, False)
                self.turn_batch_norm_off(self.Generator, True)
                for i, (_, noisy_samples, labels) in enumerate(testLoader):
                    emb = self.generate(params, noisy_inputs)
                    filename = os.path.join(embeddingDir, 'emb_{}_epoch_{}.npz'.format(i, epoch))
                    np.save(filename, emb, labels)

                self.set_requires_grad(self.Generator, True)
                self.turn_batch_norm_off(self.Generator, False)

            break


        print("Finished training!")
        print("Saving model to %s" % (params['saved_models'] + 'final_model_weights.pt'))
        torch.save(self.Generator.state_dict(), params['saved_models'] + 'final_model_weights.pt')
        self.writer.export_scalars_to_json('logger/losses.json')
        self.writer.close()

    def generate(self, params, cond_x, load=False):
        if load:
            self.Generator.load_state_dict(torch.load(params['saved_models'] + 'final_model_weights.pt'))

            self.set_requires_grad(self.Generator, False)
            self.turn_batch_norm_off(self.Generator, True)

        inputs = torch.from_numpy(cond_x)
        if self.cuda:
            inputs.cuda()
        inputs = Variable(inputs)

        emb = self.Generator.getEmbedding(inputs)
        emb = emb.cpu().numpy()

        return emb

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def scheduler(self, optimizer):
        def lambda_rule(epoch):
            lr_i = 1.0 - max(0, epoch + self.epoch_count - self.niter) / float(self.niter_decay + 1)
            return lr_i

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def turn_batch_norm_off(self, model, off=True):
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                if off:
                    layer.eval()
                else:
                    layer.train()