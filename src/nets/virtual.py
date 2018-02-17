import json

import numpy as np
from skimage.io import imread
import requests


class VirtualNet(object):

    def __init__(self, address, port=5000):
        self.address = address
        self.port = port
        self.url = 'http://{}:{}'.format(self.address, self.port)

    def predict(self, img):
        if type(img) == str:
            imgarray = imread(img)
        elif type(img) == np.ndarray:
            imgarray = img
        img2send = json.dumps({'image':imgarray.tolist(), 
                               'type':'{}'.format(imgarray.dtype)})
        r = requests.post(self.url, data=img2send)
        return json.loads(r.text)['score']