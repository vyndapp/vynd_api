from io import BytesIO
import os
from PIL import Image
import numpy as np
import requests

def get_img_from_filename(img_name: str):
    cur_dir = os.path.dirname(__file__)
    impath = os.path.join(cur_dir, 'resources/' + img_name)
    img = Image.open(impath)
    return np.asarray(img)

def url_to_img(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.asarray(img)

