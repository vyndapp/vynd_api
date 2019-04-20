import os
from PIL import Image
import numpy as np

def get_img_from_filename(img_name: str):
    cur_dir = os.path.dirname(__file__)
    impath = os.path.join(cur_dir, 'resources/' + img_name)
    img = Image.open(impath)
    return np.asarray(img)
