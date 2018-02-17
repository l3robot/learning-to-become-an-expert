import os
import time
import datetime
import json
from random import randint

import torch


class Experiment(object):

    def __init__(self, config):
        ## definitions
        self.config = dict(config)
        if 'custom_train_loss' in self.config:
            self.config['custom_train_loss'] = 'Weighted'
        else:
            self.config['custom_train_loss'] = None
        self.batch_size = config.get('batch_size', None)
        self.nb_epochs = config.get('nb_epochs', None)
        self.learning_rate = config.get('learning_rate', None)
        ## keeping timestamp
        self.timestamp = time.time()
        self.config['timestamp'] = self.timestamp
        self.ftimestamp = datetime.datetime.fromtimestamp(self.timestamp
            ).strftime('%Y-%m-%d-%H-%M-%S')
        self.root = os.path.join(self.config['results'], self.ftimestamp)
        ## create the experiment folder
        try:
            os.mkdir(self.root)
        except FileExistsError:
            self.root = os.path.join(self.config['results'], self.ftimestamp + '-' +\
                ''.join([str(randint(10)) for i in range(6)]))
        ## define names
        self.config_path = os.path.join(self.root, 'config.json')
        self.losses_path = os.path.join(self.root, 'losses.json')
        self.record_path = os.path.join(self.root, 'record.json')
        self.model_path = os.path.join(self.root, 'model.t7')
        ## saving config
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)

    def save_losses(self, losses):
        with open(self.losses_path, 'w') as f:
            json.dump(losses, f)

    def save_record(self, record):
        record2save = {}
        record2save['train_losses'] = record['train_losses']
        record2save['valid_losses'] = record['valid_losses']
        record2save['effective_epochs'] = record['effective_epochs']
        with open(self.record_path, 'w') as f:
            json.dump(record2save, f)

    def save_model(self, weights):
        if os.path.isfile(self.model_path):
            os.system("rm {}".format(self.model_path))
        torch.save(weights, self.model_path)

    def rootdir(self):
        return self.root
