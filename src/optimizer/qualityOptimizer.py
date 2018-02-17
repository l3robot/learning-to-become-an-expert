import time
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class QualityOptimizer(object):

    def __init__(self, net, config, trainset, validset, cuda=True, verbose=True):
        self.net = net
        self.batch_size = config.get('batch_size', 32)
        self.nb_epochs = config.get('nb_epochs', 10)
        self.lr = config.get('learning_rate', 0.001)
        self.loss = config.get('custom_train_loss', None)
        if self.loss is None:
            self.loss = nn.MSELoss()
        print(' [-]: USING {}'.format(self.loss))
        self.cuda = cuda
        self.verbose = verbose
        self.trainset = trainset
        self.validset = validset
        if self.cuda:
            print(' [-]: Running on GPU')
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

    def _valid(self):
        ## put in eval mode
        self.net.eval()
        ## criterion
        criterion = nn.MSELoss()
        ## put on cuda
        if self.cuda:
            inputs = Variable(self.validset.data_tensor.cuda(), volatile=True)
            targets = Variable(self.validset.target_tensor.cuda(), volatile=True)
        else:
            inputs = Variable(self.validset.data_tensor, volatile=True)
            targets = Variable(self.validset.target_tensor, volatile=True)
        ## get the score
        outputs = self.net(inputs)
        loss = criterion(outputs, targets)
        ## put in train mode
        self.net.train()
        ## return the loss
        return loss.data.cpu().numpy()[0]

    def _train_one_step(self, data):
        ## criterion and data
        criterion = self.loss
        inputs, targets = data
        ## put on cuda
        if self.cuda:
            targets = targets.cuda()
            inputs = inputs.cuda()
        ## create variables
        inputs, targets = Variable(inputs), Variable(targets)
        ## optimize
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        ## get the loss
        loss_numpy = loss.data.cpu().numpy()[0]
        ## delete heavy structures
        del outputs, loss, inputs, targets
        ## return the loss
        return loss_numpy

    def _train_one_epoch(self, trainloader):
        batch_train_losses = []
        for i, data in enumerate(trainloader):
            ## do a step
            train_loss = self._train_one_step(data)
            ## save the loss
            batch_train_losses.append(train_loss)
        return np.mean(batch_train_losses)

    def _train_with_valid(self):
        ## put the net in train mode
        self.net.train()
        ## put the net on cuda
        if self.cuda:
            self.net.cuda()
        ## create the trainloader
        trainloader = torch.utils.data.DataLoader(self.trainset, self.batch_size, shuffle=True)
        ## inform the user
        if self.verbose:
            print(' [-]: Beginning training for max {} epochs with SGD({}, {})'.format(
                self.nb_epochs, self.batch_size, self.lr))
        ## begin the training
        effective_epochs = self.nb_epochs
        min_valid_loss = np.inf
        train_losses = []
        valid_losses = []
        worse_counter = 0
        for epoch in range(self.nb_epochs):
            ## train one epoch 
            train_loss = self._train_one_epoch(trainloader)
            ## get the valid score
            valid_loss = self._valid()
            ## add losses to history
            train_losses.append(float(train_loss))
            valid_losses.append(float(valid_loss))
            ## keep the best minimum
            if valid_loss < min_valid_loss:
                worse_counter = 0
                effective_epochs = epoch
                min_train_loss = train_loss
                min_valid_loss = valid_loss
                best_weights = copy.deepcopy(self.net.state_dict())
            else:
                worse_counter += 1
                if worse_counter == 10:
                    if self.verbose:
                        print(' [-]: * Early stopping *')
                    break
            # inform the user
            if self.verbose:
                print(' [-]: epoch: {:3}, train-loss: {:.6f}, valid-loss {:.6f}, min-valid-loss {:.6f}'\
                    .format(epoch + 1, train_loss, valid_loss, min_valid_loss))
        ## inform the user
        if self.verbose:
            print(' [-]: Training finished after {} epochs with a valid loss of : {}'.format(
                   effective_epochs + 1, min_valid_loss))
        ## remove from cuda
        if self.cuda:
            self.net.cpu()
        ## put the net in eval mode
        self.net.eval()
        ##
        training_record = {'train_losses': {'list': train_losses, 'min': float(min_train_loss)},
                           'valid_losses': {'list': valid_losses, 'min': float(min_valid_loss)},
                           'best_weights': best_weights, 'effective_epochs': effective_epochs}
        ## return losses
        return training_record

    def _train(self):
        ## put the net in train mode
        self.net.train()
        ## put the net on cuda
        if self.cuda:
            self.net.cuda()
        ## create the trainloader
        trainloader = torch.utils.data.DataLoader(self.trainset, self.batch_size, shuffle=True)
        ## inform the user
        if self.verbose:
            print(' [-]: Beginning training for {} epochs with SGD({}, {})'.format(
                self.nb_epochs, self.batch_size, self.lr))
        ## begin the training
        for epoch in range(self.nb_epochs):
            ## train one epoch 
            train_loss = self._train_one_epoch(trainloader)
            ## inform the user
            if self.verbose:
                print(' [-]: epoch: {}, train-loss: {:.3f}'.format(epoch + 1, train_loss))
        ## inform the user
        if self.verbose:
            print(' [-]: Training finished')
        ## remove from cuda
        if self.cuda:
            self.net.cpu()
        ## put the net in eval mode
        self.net.eval()
        ## return losses
        return train_loss

    def train(self):
        if self.validset is None:
            return self._train()
        else:
            return self._train_with_valid()

    def reset(self):
        self.net.init()
