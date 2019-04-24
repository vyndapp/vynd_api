from io import BytesIO
from pathlib import Path
from PIL import Image

import numpy as np
import requests
import base64
import os

def get_img_from_filename(img_name: str):
    cur_dir = os.path.dirname(__file__)
    impath = Path(cur_dir, 'resources/' + img_name)#os.path.join(cur_dir, 'resources\\' + img_name)
    img = Image.open(impath)
    return np.asarray(img)

def url_to_img(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.asarray(img)

def url_to_base64(url: str):
    response_content = requests.get(url).content
    base64_img = base64.b64encode(response_content)
    return base64_img