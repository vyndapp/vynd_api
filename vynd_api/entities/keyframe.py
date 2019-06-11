"""Canonical representations for images being handled"""

from typing import Optional

import json
import numpy as np


from ..utils import image_utils, numpy_encoder
from ..data.db_utils import np_to_binary, binary_to_b64

class KeyFrame:

    def __init__(self, keyframe_image, timestamp: Optional[int] = 0):
        """
        Params:
        - keyframe_image: numpy.ndarray | bytes
        - timestamps: Optional[int]=0
        """
        self.__timestamp = timestamp
        self.__set_image(keyframe_image)
        self.__set_base64_image(keyframe_image)

    def __save_json_image(self, keyframe_image):
        if(keyframe_image.all() != None):
            return json.dumps(keyframe_image, cls=numpy_encoder.NumpyEncoder)
        else:
            return None
    
    def __process_keyframe_image(self, keyframe_image):
        if isinstance(keyframe_image, bytes): # image passed in base64 representation
            return image_utils.base64_to_rgb(keyframe_image)
        elif isinstance(keyframe_image, np.ndarray): # image passed in numpy array representation
            return keyframe_image
        else:
            return None

    def __get_image(self) -> np.ndarray:
        return self.__image

    def __set_image(self, image):
        self.__image = self.__process_keyframe_image(image)

    def __get_base64_image(self):
        return self.__base64_image
    
    def __set_base64_image(self, image):
        if isinstance(image, bytes):
            self.__base64_image = image
        elif isinstance(image, np.ndarray):
            self.__base64_image = binary_to_b64(np_to_binary(image))
        else:
            self.__base64_image = None

    @property
    def timestamp(self):
        return self.__timestamp

    image: np.ndarray = property(__get_image, __set_image)
    base64_image = property(__get_base64_image, __set_base64_image)
