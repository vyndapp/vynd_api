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

def rgb_to_base64(image):
    return base64.b64encode(image)

def resize_image(image, new_shape=(96, 96)):
    return cv2.resize(image, new_shape) 