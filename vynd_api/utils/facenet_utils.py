
import os
import numpy as np

from pathlib import Path

from keras.models import model_from_json

def load_model(model_name: str):
    """
    Loads a Keras model from the JSON and H5 files
    """
    cur_dir = os.path.dirname(__file__)
    models_path = '../models/'
    
    json_model_path = Path(cur_dir, models_path + model_name + '.json')
    weights_model_path = Path(cur_dir, models_path + model_name + '.h5')

    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weights_model_path)
    return loaded_model

def normalize_image(image, channels_first = True):
    """
    Normalize an image so that the values become in range from 0 to 1 (inclusive)
    """
    if(channels_first == True): 
        channels_order = (2, 0, 1)
    else:
        channels_order = (0, 1, 2)
    return np.around(np.transpose(image, channels_order)/255.0, decimals=12)