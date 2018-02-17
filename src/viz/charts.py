import os

import numpy as np
import matplotlib.pyplot as plt


def plot_train_valid_curve(training_record, rootdir=None, loss_str='RMSE'):
    plt.close('all')
    train_losses = training_record['train_losses']['list']
    valid_losses = training_record['valid_losses']['list']
    if loss_str == 'RMSE':
        train_losses = np.sqrt(train_losses)
        valid_losses = np.sqrt(valid_losses)
    effective_epochs = training_record['effective_epochs']
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(valid_losses)), valid_losses)
    plt.axvline(effective_epochs, color='g', linestyle='solid')
    plt.title('train losses compared to valid losses')
    plt.xlabel('epochs')
    plt.ylabel('losses ({})'.format(loss_str))
    plt.legend(['train losses', 'validation losses'])
    if rootdir is not None:
        plt.savefig(os.path.join(rootdir, 'train_valid_curve.png'))


def plot_split_histogram(train_targets, valid_targets, test_targets, rootdir=None):
    plt.close('all')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    bins = np.linspace(0.0, 1.0, 30)
    ax1.hist(train_targets, bins)
    ax1.set_title('train split')
    n1, _, _ = ax2.hist(valid_targets, bins)
    ax2.set_title('valid split')
    n2, _, _ = ax3.hist(test_targets, bins)
    ax3.set_title('test split')
    ax2.set_ylim([0.0, np.max(np.concatenate((n1, n2))) + 2])
    ax3.set_ylim([0.0, np.max(np.concatenate((n1, n2))) + 2])
    if rootdir is not None:
        plt.savefig(os.path.join(rootdir, 'split_histogram.png'))

