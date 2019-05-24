
from bson.binary import Binary

from ..utils.numpy_encoder import NumpyEncoder

import json
import pickle

def np_to_binary(nparray):
    return Binary(pickle.dumps(nparray, protocol=2), subtype=128)

def np_to_json(nparray):
    return json.dumps(nparray, cls=NumpyEncoder)

def binary_to_np(binary):
    return pickle.loads(binary)
