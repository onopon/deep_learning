# !/usr/bin/env python
# For Example.
# $python execute.py http://livedoor.blogimg.jp/minatoku_sposen/imgs/0/6/063bb318.png
# return int number(range of 0 - 2)
# 0: available.
# 1: a part of available.
# 2: disavailable.

import sys
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
import urllib3
import io
from urllib.request import urlopen

WIDTH = 64
HEIGHT = 64
CROP_X = 310 
CROP_Y = 240
CROP_WIDTH = 100
CROP_HEIGHT = 110
CHANNEL = 3

if __name__ == '__main__':
    args = sys.argv
    img_url = args[1]
    model_path = args[2] if len(args) == 3 else ('./minatoku_sc20180620013645.hdf5')
    model = load_model(model_path)
    file = io.BytesIO(urlopen(img_url).read())
    img = Image.open(file)
    img = img.crop((CROP_X, CROP_Y, CROP_X + CROP_WIDTH, CROP_Y + CROP_HEIGHT)).resize((WIDTH, HEIGHT))
    im = np.array(img)
    im = im[...,:CHANNEL]
    test_x = np.array([im])
    pred_y = model.predict(test_x)
    print(np.argmax(pred_y))
