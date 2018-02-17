"""
nets.qualityNet: defines the nets structure
"""
import math

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 5,
                     stride=stride, padding=2)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3,
                     stride=stride, padding=1)

####
####
class QualityNetBase(nn.Module):

    def __init__(self, channels=1):
        super(QualityNetBase, self).__init__()

    def conv_out_(self, x, size=2):
        return F.max_pool2d(F.elu(x), (size, size), ceil_mode=True)

    def linear_out_(self, x):
        return F.elu(x)

    def init(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, X):
        y = self.linear_parts(self.conv_parts(X))
        return F.sigmoid(y)

    def loading(self, PATH):
        self.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
        return self

    def _pre_process(self, X, cuda=False):
        if type(X) == np.ndarray:
            net_input = torch.from_numpy(X).float()
        else:
            net_input = X.clone()
        if cuda:
            net_input = net_input.cuda()
        net_input = Variable(net_input, volatile=True)
        return net_input

    def vectorize(self, X):
        return self.linear_parts_vec(self.conv_parts(X))

    def neuron2vec(self, X, batch_size=100, cuda=False):
        outputs = []
        for i in range(math.ceil(len(X) / batch_size)):
            net_input = self._pre_process(X[i*batch_size:i*batch_size+batch_size], cuda)
            if cuda:
                self.cuda()
            output = self.vectorize(net_input)
            outputs.append(output.data.cpu())
            if cuda:
                self.cpu()
            del net_input, output
        return torch.cat(outputs, dim=0)

    def predict(self, X, cuda=False):
        net_input = self._pre_process(X, cuda)
        if cuda:
            self.cuda()
        output = self.forward(net_input)
        output = output.data.cpu().numpy().flatten()
        if cuda:
            self.cpu()
        return output

###
###
class QualityNet(QualityNetBase):

    def __init__(self, channels=1):
        super(QualityNet, self).__init__()
        ## Convolution layers
        self.conv32 = conv3x3(channels, 32)
        self.conv64 = conv3x3(32, 64)
        self.conv128 = conv3x3(64, 128)
        self.conv192 = conv3x3(128, 192)
        self.conv256 = conv3x3(192, 256)
        self.conv320 = conv3x3(256, 320)
        ## BatchNorm layers
        self.bn2d64 = nn.BatchNorm2d(64)
        self.bn2d128 = nn.BatchNorm2d(128)
        self.bn2d192 = nn.BatchNorm2d(192)
        self.bn2d256 = nn.BatchNorm2d(256)
        self.bn2d320 = nn.BatchNorm2d(320)
        self.bn1d512 = nn.BatchNorm1d(512)
        self.bn1d256 = nn.BatchNorm1d(256)
        ## Fully Connected layers
        self.fc512 = nn.Linear(320 * 4 * 4, 512)
        self.fc256 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 1)

    def conv_parts(self, X):
        ## Convolution part
        y = self.conv_out_(self.conv32(X))
        y = self.conv_out_(self.bn2d64(self.conv64(y)))
        y = self.conv_out_(self.bn2d128(self.conv128(y)))
        y = self.conv_out_(self.bn2d192(self.conv192(y)))
        y = self.conv_out_(self.bn2d256(self.conv256(y)))
        y = self.conv_out_(self.bn2d320(self.conv320(y)))
        return y

    def linear_parts(self, X):
        ## Flattening
        y = X.view(-1, 320 * 4 * 4)
        ## Fully connected part
        y = self.linear_out_(self.bn1d512(self.fc512(y)))
        y = F.dropout(y, 0.2, training=self.training)
        y = self.linear_out_(self.bn1d256(self.fc256(y)))
        y = F.dropout(y, 0.2, training=self.training)
        return self.fc1(y)

    def linear_parts_vec(self, X):
        ## Flattening
        y = X.view(-1, 320 * 4 * 4)
        ## Fully connected part
        y = self.linear_out_(self.bn1d512(self.fc512(y)))
        y = F.dropout(y, 0.2, training=self.training)
        y = self.linear_out_(self.bn1d256(self.fc256(y)))
        y = F.dropout(y, 0.2, training=self.training)
        return y







