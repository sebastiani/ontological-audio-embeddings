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
from data_loader.AudioSetDataset import AudioSetDataset
# torch.backends.cudnn.enabled = False


class BaseModel(object):
    def __init__(self, params):
        self.model = None
        self.cuda = params['cuda']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.momentum = params['momentum']

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
            self.model = self.model.cuda()

        loss_fn = nn.CrossEntropyLoss()
        if self.cuda:
            loss_fn = loss_fn.cuda()

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # remember to set volatile=True for validation input and label
        for epoch in range(self.epochs):
            train_running_loss = 0.0
            validation_running_loss = 0.0
            for i, (samples, labels) in enumerate(trainLoader):
                inputs, labels = torch.from_numpy(samples), torch.from_numpy(labels).long()

                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = self.model(inputs.float())

                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.cpu().item()

                torch.no_grad()
                self.model.eval()
                eval_end = False
                for j, (vals, vlabels) in range(numValBatches):
                    val_inputs, v_labels = torch.from_numpy(vals), torch.from_numpy(vlabels)

                    if self.cuda:
                        val_inputs, v_labels = val_inputs.cuda(), v_labels.cuda()

                    val_inputs, v_labels = Variable(val_inputs), Variable(v_labels)
                    val_outputs = self.model.forward(val_inputs.float())
                    val_loss = loss_fn(val_outputs, v_labels)
                    validation_running_loss += val_loss.cpu().item()
                    if eval_end:
                        break

                self.model.train()

                if i % 5 == 0:
                    print('[%d, %5d] train loss: %.3f' %
                          (epoch + 1, i + 1, train_running_loss / 100))
                    train_running_loss = 0.0

                    print('[%d, %5d] val loss: %.3f' %
                          (epoch + 1, i + 1, validation_running_loss / 100))
                    validation_running_loss = 0.0

                # ADD CHECKPOINTING HERE

        print("Finished training!")
        print("Saving model to %s" % (params['saved_models'] + 'model_weights.pt'))
        torch.save(self.model.state_dict(), params['saved_models'] + 'model_weights.pt')

    def predict(self, params):
        self.model.load_state_dict(torch.load(params['saved_models'] + 'model_weights.pt'))
        self.model.eval()
        raise ("Not implemented yet")
