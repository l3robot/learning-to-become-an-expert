import os
import time
import random

import numpy as np
from skimage.io import imread, imsave

from .utils import img_to_float


def data_augmentation(X, y, flip_ratio=0.5):
    nb_flip = int(flip_ratio * len(X))
    nb_rotate = len(X) - nb_flip
    
    XX = X.copy()

    idx_flip = np.zeros(len(X)).astype(np.bool)
    while idx_flip.sum() < nb_flip:
        a = np.random.randint(len(X))
        if idx_flip[a] != True:
            idx_flip[a] = True

    XX[idx_flip, 0] = np.flip(XX[idx_flip, 0], 1)
    idx_rotate = np.where(idx_flip == False)[0]
    rotations = np.random.randint(1, 4, nb_rotate)
    XX[idx_rotate, 0] = np.array([np.rot90(XX[idx_rotate[i], 0], rotations[i]) for i in range(nb_rotate)])

    return np.concatenate((X, XX), axis=0), np.concatenate((y, y), axis=0)

def _load_dataset_of_type(path, img_type):
    type_root = os.path.join(path, img_type)
    type_path = os.listdir(type_root)
    type_path.sort()
    type_path = np.array(list(filter(lambda x: os.path.splitext(x)[-1] == '.tiff', type_path)))
    names = np.array(list(map(lambda x: '-'.join(os.path.splitext(x)[0].split('-')[:-1]), type_path)))
    y = np.array(list(map(lambda x: float(os.path.splitext(x)[0].split('-')[-1]), type_path))).astype(np.float32)
    type_path = list(map(lambda x: os.path.join(type_root, x), type_path))
    X = np.array(list(map(lambda x: img_to_float(imread(x)), type_path)))
    return X, y, names

def _load_dataset(path, verbose, return_names=False, return_confocals=False):
    start = time.time()

    to_return = []

    X, y, names = _load_dataset_of_type(path, 'sted')
    to_return.extend([X, y])

    if return_names:
        to_return.append(names)

    if return_confocals:
        Xc, yc, _ = _load_dataset_of_type(path, 'confocal')
        to_return.append(Xc)

    if verbose:
        print(' [-]: It took {}s to load the dataset'.format(time.time() - start))
        print(' [-]: It contains {} examples'.format(len(X)))

    return tuple(to_return)

def load_dataset(path, verbose=False, return_names=False, return_confocals=False):
    return _load_dataset(path, verbose, return_names, return_confocals)


def draw_from(X, y, k):
    real_k = min(len(X), k)
    idx = random.sample(range(len(X)),k=real_k)
    return X[idx], y[idx]

def balance_dataset(X, y, verbose=False):
    hist, bins = np.histogram(y, bins=2, range=(0,1))
    nb = int(np.mean(hist[:len(hist)//2]))
    X_acc, y_acc = [], []
    for bin_low, bin_high in zip(bins[:-1], bins[1:]):
        mask = (y > bin_low) & (y < bin_high)
        xx, yy = draw_from(X[mask], y[mask], nb)
        X_acc.append(xx); y_acc.append(yy)
    X_new = np.concatenate(X_acc, axis=0)
    y_new = np.concatenate(y_acc, axis=0)
    if verbose:
        print(' [-]: train dataset reduced to {} examples'.format(len(X_new)))
    return X_new, y_new


