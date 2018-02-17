# QualityNet
### Learning to Become an Expert: Deep Networks Applied To Super-Resolution Microscopy

#### source code

---

### Abstract 

With super-resolution optical microscopy, it is now possible to observe molecular interactions in living cells. The obtained images have a very high spatial precision but their overall quality can vary a lot depending on the structure of interest and the imaging parameters. Moreover, evaluating this quality is often difficult for non-expert users. In this work, we tackle the problem of learning the quality function of super-resolution images from scores provided by experts. More specifically, we are proposing a system based on a deep neural network that can provide a quantitative quality measure of a STED image of neuronal structures given as input. We conduct a user study in order to evaluate the quality of the predictions of the neural network against those of a human expert. Results show the potential while highlighting some of the limits of the proposed approach.

### Dependencies
- Install [pytorch](http://pytorch.org/)
- Install these python packages:
```shell
pip install numpy scipy scikit-learn scikit-image request
```

### Training a network
Use the script ``python train.py``:
```shell
usage: train.py [-h] [--test-ratio ratio] [--valid-ratio ratio] [-bs size]
                [-ep nb] [-lr rate] [-rs state] [--balanced] [--cuda] [-v]
                data results

Training the qualityNet

positional arguments:
  data                              path of data
  results                           path of results

optional arguments:
  -h, --help                        show this help message and exit
  --test-ratio ratio                ratio of the data for testing (default: 0.1)
  --valid-ratio ratio               ratio of the data for validation (default: 0.1)
  -bs size, --batch-size size       SGD batch size (default: 16)
  -ep nb, --nb-epochs nb            SGD number of epochs (default: 5)
  -lr rate, --learning-rate rate    SGD learning rate (default: 0.001)
  -rs state, --random-state state   random state of train/valid/test split (default: 42)
  --balanced                        trying to balance data or not (default: False)
  --cuda                            use GPU or not (default: False)
  -v, --verbose                     print information (default: False)
```

### Test the network
Use the script ``python test.py``:
```shell
usage: test.py [-h] [--virtual] [--url URL] [--port PORT] [--xp XP] [--cuda]
               [-v] data

Testing the qualityNet

positional arguments:
  data                              path of data

optional arguments:
  -h, --help                        show this help message and exit
  --virtual                         using virtual net (default: False)
  --url URL                         url if using virtual net (default: 0.0.0.0)
  --port PORT                       port if using virtual net (default: 5000)
  --xp XP                           xp path if not using virtual net (default: .)
  --cuda                            use GPU or not (default: False)
  -v, --verbose                     print information (default: False)
```

### Starting a network server
Use the script ``python server.py xp``, where ``xp`` is the path of the net experiment folder.  
