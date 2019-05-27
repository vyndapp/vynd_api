
from bson.binary import Binary
from PIL import Image
from io import BytesIO

from ..utils.numpy_encoder import NumpyEncoder

import base64
import json
import pickle

def np_to_binary(nparray):
    return Binary(pickle.dumps(nparray, protocol=2), subtype=128)

def np_to_json(nparray):
    return json.dumps(nparray, cls=NumpyEncoder)

def binary_to_np(binary):
    return pickle.loads(binary)

def binary_to_b64(binary):
    pil_img = Image.fromarray(binary_to_np(binary), 'RGB')
    buffer = BytesIO()
    pil_img.save(buffer, 'JPEG')
    image = buffer.getvalue()
    b64 = base64.b64encode(image)
    pil_img.close()
    return b64
