""" helper functions for parsing images from one format to another """

import base64
import numpy as np

def base64_to_rgb(base64_text):
    img = base64.decodebytes(base64_text)
    return np.asarray(img)
