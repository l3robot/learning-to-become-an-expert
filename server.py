import sys
import os
import json

import numpy as np
from skimage.external.tifffile import imread
from flask import Flask
from flask import render_template, request

from src.dataset import img_to_float
from src.nets import load_model


if len(sys.argv) < 2:
    print(' [!] you need to give an xp repository')
    exit()

XP = sys.argv[1]
with open(os.path.join(XP, 'config.json')) as f:
    config = json.load(f)
MEAN = config['mean']
STD = config['std']
## creating app
app = Flask(__name__)
## loading model
net = load_model(os.path.join(XP, 'model.t7'))

@app.route("/", methods=['POST'])
def get_score():
    data = json.loads(request.data.decode('utf-8'))
    img = np.array(data['image']).astype(data['type'])
    img = (img[np.newaxis, np.newaxis, :, :] - MEAN) / STD
    score = float(net.predict(img)[0])
    return json.dumps({'score':score}), 200, {'ContentType':'application/json'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)