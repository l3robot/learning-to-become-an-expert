import os
import json
import random
import argparse

import numpy as np

from src.dataset import load_dataset
from src.dataset import normalize_data, create_split
from src.nets import load_model, VirtualNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the qualityNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--virtual', help='using virtual net', action='store_true', default=False)
    parser.add_argument('--url', help='url if using virtual net', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='port if using virtual net', type=int, default=5000)
    parser.add_argument('--xp', help='xp path if not using virtual net', type=str, default='.')
    parser.add_argument('data', help='path of data', type=str)
    parser.add_argument('--cuda', help='use GPU or not', action="store_true",
                        default=False)
    parser.add_argument('-v', '--verbose', help='print information', action="store_true",
                        default=False)
    args = parser.parse_args()

    ## loading data for the example
    data, targets = load_dataset(args.data, verbose=args.verbose)

    if args.virtual:
        ## here I don't assume the split is the same as the training one!
        _, _, test = create_split(data, targets)
        test_data, test_target = test
        idx = list(range(len(test_data)))
        random.shuffle(idx)
        net = VirtualNet(args.url, args.port)
        for i in idx[:10]:
            score = net.predict(test_data[i])
            print(' [-] prediction: {:.3f}, true label: {:.3f}'.format(score, test_target[i]))
    else:
        XP = args.xp
        with open(os.path.join(XP, 'config.json')) as f:
            config = json.load(f)
        mean = config['mean']
        std = config['std']
        random_state = config['random_state'] 

        ## here I impose the split to be the same as the training one!
        _, _, test = create_split(data, targets, random_state=random_state)

        test_data, test_target = test
        test_data = normalize_data(test_data, mean, std)

        idx = list(range(len(test_data)))
        random.shuffle(idx)
        net = load_model(os.path.join(XP, 'model.t7'))
        for i in idx[:10]:
            score = net.predict(test_data[i][np.newaxis, np.newaxis, :, :], cuda=args.cuda)[0]
            print(' [-] prediction: {:.3f}, true label: {:.3f}'.format(score, test_target[i]))