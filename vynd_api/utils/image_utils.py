""" helper functions for parsing images from one format to another """

import base64
import cv2
import numpy as np

from PIL import Image
from io import BytesIO

def base64_to_rgb(base64_text):
    # todo: make sure that the conversion takes place
    img = base64.decodebytes(base64_text)
    pil_img = Image.open(BytesIO(img))
    return np.asarray(pil_img)

def normalize_image(image, channels_first = True):
    if(channels_first == True): 
        channels_order = (2, 0, 1)
    else:
        channels_order = (0, 1, 2)
    return np.around(np.transpose(image, channels_order)/255.0, decimals=12)

def resize_image(image, new_shape=(96, 96)):
    return cv2.resize(image, new_shape) 