
import os

from pathlib import Path

from keras.models import model_from_json

def load_model(model_name: str):
    cur_dir = os.path.dirname(__file__)
    models_path = '../models/'
    
    json_model_path = Path(cur_dir, models_path + model_name + '.json')
    weights_model_path = Path(cur_dir, models_path + model_name + '.h5')

    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weights_model_path)
    return loaded_model