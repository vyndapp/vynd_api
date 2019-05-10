from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List

import numpy as np
import requests
import base64
import os

def get_img_from_filename(filename: str):
    cur_dir = os.path.dirname(__file__)
    impath = Path(cur_dir, filename)#os.path.join(cur_dir, 'resources\\' + img_name)
    # img = Image.open(impath)
    with Image.open(impath) as img:
        return np.asarray(img)

def get_all_local_images(dir: str) -> List[np.ndarray]:
    cur_dir = os.path.dirname(__file__)
    cur_path = Path(cur_dir, dir)
    directory = os.fsencode(cur_path)

    local_images = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".JPG"):
            local_images.append(get_img_from_filename(dir + filename))
            
    return local_images

def get_all_cropped_images() -> List[np.ndarray]:
    cur_dir = os.path.dirname(__file__)
    cur_path = Path(cur_dir, 'detected_faces/')
    directory = os.fsencode(cur_path)

    cropped_images = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            cropped_images.append(get_img_from_filename('detected_faces/' + filename))
            
    return cropped_images

def save_img(dir:str, image: np.array, img_name: str):
    cur_dir = os.path.dirname(__file__)
    impath = Path(cur_dir, dir + img_name)
    img = Image.fromarray(image)
    img.save(impath)

def url_to_img(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.asarray(img)

def url_to_base64(url: str):
    response_content = requests.get(url).content
    base64_img = base64.b64encode(response_content)
    return base64_img