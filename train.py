#!/usr/bin/env python

import os
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.utils import shuffle
import torch
import torchvision

from src.nets import QualityNet
from src.dataset import load_dataset, data_augmentation 
from src.dataset import create_split, normalize_data
from src.optimizer import QualityOptimizer
from src.experiment import Experiment
from src.viz import plot_split_histogram, plot_train_valid_curve

torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the qualityNet')

    parser.add_argument('--test-ratio', help='ratio of the data for testing', type=float, 
                        default=0.1)
    parser.add_argument('--valid-ratio', help='ratio of the data for validation', type=float, 
                        default=0.1)
    parser.add_argument('-bs', '--batch-size', type=int, help='SGD batch size',
                        default=16)
    parser.add_argument('-ep', '--nb-epochs', type=int, help='SGD number of epochs',
                        default=5)
    parser.add_argument('-lr', '--learning-rate', type=float, help='SGD learning rate', 
                        default=0.001)
    parser.add_argument('-rs', '--random-state', type=int, help='random state of train/valid/test split',
                        default=42)
    parser.add_argument('--balanced', help='trying to balance data or not',
                        action='store_true', default=False)
    parser.add_argument('--cuda', help='use GPU or not', action="store_true",
                        default=False)
    parser.add_argument('-v', '--verbose', help='print information', action="store_true",
                        default=False)
    parser.add_argument('data', help='path of data', type=str)
    parser.add_argument('results', help='path of results', type=str)
    
    args = parser.parse_args()

    ## fill up config dict
    config = {'type':'qualityNet', 
              'random_state': args.random_state,
              'batch_size': args.batch_size,
              'nb_epochs': args.nb_epochs,
              'learning_rate': args.learning_rate,
              'data': args.data,
              'results': args.results}

    ## net creation and data loading
    net = QualityNet()
    data, targets = load_dataset(args.data, verbose=args.verbose)

    ## data selection
    split_config = [args.valid_ratio, args.test_ratio, args.random_state]
    train, valid, test = create_split(data, targets, *split_config)
    train_data, train_targets = train[0][:,np.newaxis,:,:], train[1]
    if args.balanced:
        train_data, train_targets = balance_dataset(train_data, train_targets, args.verbose)
    valid_data, valid_targets = valid[0][:,np.newaxis,:,:], valid[1]
    if test is not None:
        test_data, test_targets = test[0][:,np.newaxis,:,:], test[1]

    ## inform the user
    if args.verbose:
        print(' [-]: using {} training examples (~{}%)'.format(len(train_data), 
              int(100*len(train_data)/len(data))))
        print(' [-]: using {} validation examples (~{}%)'.format(len(valid_data), 
              int(100*len(valid_data)/len(data))))
        if test is not None:
            print(' [-]: using {} testing examples (~{}%)'.format(len(test_data), 
                  int(100*len(valid_data)/len(data))))
        else:
            print(' [-]: using no testing examples (0%)')

    ## Normalization
    mean, std = np.mean(train_data), np.std(train_data)
    config['mean'], config['std'] = float(mean), float(std)
    train_data = normalize_data(train_data, mean, std)
    if args.balanced:
        train_data, train_targets = balance_dataset(train_data, train_targets, 
                                                    args.verbose)
    valid_data = normalize_data(valid_data, mean, std)
    if test is not None:
        test_data = normalize_data(test_data, mean, std)

    ## setup the environment folder
    xp = Experiment(config)

    ## save split histrogram
    if test is not None:
        plot_split_histogram(train_targets, valid_targets, test_targets, rootdir=xp.rootdir())

    ## data augmentation
    train_data, train_targets = data_augmentation(train_data, train_targets)
    if args.verbose:
        print(' [-]: data augmentation to {} training examples'.format(len(train_data)))
        print()

    ## pytorch dataset creation
    train_data = torch.from_numpy(train_data).float()
    valid_data = torch.from_numpy(valid_data).float()
    train_targets = torch.from_numpy(train_targets).float()
    valid_targets = torch.from_numpy(valid_targets).float()
    trainset = torch.utils.data.TensorDataset(train_data, train_targets)
    validset = torch.utils.data.TensorDataset(valid_data, valid_targets)

    ## trainer creation
    trainer = QualityOptimizer(net, config, trainset, validset, args.cuda, args.verbose)

    ## inform the user
    print(' [-]: Beginning training')

    ## training
    losses = {}
    ## train the model
    training_record = trainer.train()
    train_loss = training_record['train_losses']['min']
    valid_loss = training_record['valid_losses']['min']
    best_weights = training_record['best_weights']
    ## saving the weights
    xp.save_record(training_record)
    xp.save_model(best_weights)
    plot_train_valid_curve(training_record, rootdir=xp.rootdir())
    ## updating the loss
    losses['train-loss'] = float(train_loss)
    losses['valid-loss'] = float(valid_loss)

    ## inform the user
    print(' [-]: Final minimal valid RMSE : {}'.format(np.sqrt(losses['valid-loss'])))
    print()

    ## saving
    ## load the previous best model
    net = QualityNet()
    net.loading(xp.model_path)
    net.eval()
    ## test
    if test is not None:
        predictions = net.predict(test_data)
        test_loss = np.mean((test_targets - predictions) ** 2)
        test_loss_std = np.std((test_targets - predictions) ** 2)
        losses['test-loss'] = float(test_loss)
        losses['test-loss-std'] = float(test_loss_std)
        ## inform the user
        print(' [-]: Final minimal test RMSE : {}'.format(np.sqrt(losses['test-loss'])))
        print(' [-]: Final minimal test std RMSE : {}'.format(np.sqrt(losses['test-loss-std'])))
        print()
    ## save losses
    xp.save_losses(losses)



