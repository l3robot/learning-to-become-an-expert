import numpy as np
from sklearn import utils as skutils
import torch


def img_to_float(X):
    if X.dtype == np.uint16:
        return rescale_data(X, float(2**16-1), 0.0)
    elif X.dtype == np.int16:
        return rescale_data(X, float(2**15-1), -float(2**15))
    elif X.dtype == np.uint8:
        return rescale_data(X, float(2**8-1), 0.0)
    elif X.dtype == np.int8:
        return rescale_data(X, float(2**7-1), -float(2**7))
    else:
        raise TypeError

def saturate_data(X, saturation):
    X_saturate = X.copy()
    upper_limit = np.min(X) + saturation * (np.max(X) - np.min(X))
    X_saturate[X > upper_limit] = upper_limit
    X_saturate = (X_saturate - np.min(X_saturate)) / (np.max(X_saturate) - np.min(X_saturate))
    return X_saturate

def rescale_data(X, X_max, X_min):
    return (X - X_min) / (X_max - X_min)

def normalize_data(data, mean, std):
    return (data - mean) / std

def denormalize_data(data, mean, std):
    return (data * std) + mean

def create_split(data, target, valid_ratio=0.1, test_ratio=0.1, random_state=42):
    X, y = skutils.shuffle(data, target, random_state=random_state)
    test_separator = int(test_ratio * len(data))
    valid_separator = int(valid_ratio * len(data)) + test_separator
    if test_separator > 0:
        train = X[:-valid_separator], y[:-valid_separator]
        valid = X[-valid_separator:-test_separator], y[-valid_separator:-test_separator]
        test = X[-test_separator:], y[-test_separator:]
    else:
        train = X[:-valid_separator], y[:-valid_separator]
        valid = X[-valid_separator:], y[-valid_separator:]
        test = None
    return train, valid, test

def train_test_split(data, target, test_ratio=0.1, random_state=42):
    X, y = skutils.shuffle(data, target, random_state=random_state)
    test_separator = int(test_ratio * len(data))
    train = X[:-test_separator], y[:-test_separator]
    test = X[-test_separator:], y[-test_separator:]
    return train, test

def uniform_balanced_weights(target, bins):
    count = np.histogram(target, bins, density=False)[0]
    weights = count / np.sum(count)
    weights[weights != 0] = 1 / weights[weights != 0]
    return weights

def histogram_equilizer(target, L):
    bins = np.linspace(0.0, 1.0, L+1)
    hist = np.histogram(target, bins, density=True)[0]
    step = (bins[1:]-bins[:-1])[0]
    hist = hist * step
    cumul = np.cumsum(hist)
    new_target = []
    for t in target:
        idx = np.where(bins >= t)[0][0]
        new_target.append(cumul[max(0, idx-1)])
    return new_target

def create_pairs(size, shuffle=False, random_state=None):
    x1, x2 = np.mgrid[:size, :size]
    if shuffle:
        x1, x2 = skutils.shuffle(x1.ravel(), x2.ravel(), random_state=random_state)
    else:
        x1, x2 = x1.ravel(), x2.ravel()
    pairs = np.array([x for x in zip(x1, x2) if x[0] != x[1]])
    assert len(pairs) == size * (size - 1)
    return pairs

def rbf(y, sigma, use_torch=True):
    if use_torch:
        return torch.exp(-((y[:,0] - y[:,1]) ** 2) / (2 * sigma ** 2))
    else:
        return np.exp(-((y[:,0] - y[:,1]) ** 2) / (2 * sigma ** 2))

def smooth_target(y, sigma, use_torch=True):
    s = rbf(y, sigma, use_torch)
    if use_torch:
        s = s.unsqueeze(1)
        y_hat = torch.zeros(y.size())
        y_hat[torch.arange(0, len(y)).long(), torch.max(y, dim=1)[1]] = (1 - s) + s / 2
        y_hat[torch.arange(0, len(y)).long(), torch.min(y, dim=1)[1]] = s / 2
    else:
        y_hat = np.zeros(y.shape)
        y_hat[np.arange(0, len(y)), np.argmax(y, axis=1)] = (1 - s) + s / 2
        y_hat[np.arange(0, len(y)), np.argmin(y, axis=1)] = s / 2
    return y_hat