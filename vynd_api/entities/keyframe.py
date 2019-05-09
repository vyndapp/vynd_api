"""Canonical representations for images being handled"""

import numpy as np
import json

from ..utils import image_utils, numpy_encoder
from ..facedetection.face_detection_results import FaceDetectionResults

# frame timestamp can be added later as an attribute
class KeyFrame:
    """
    image: np.ndarray,
    face_detection_results: FaceDetectionResults
    """
    
    __face_detection_results: FaceDetectionResults

    def __init__(self, keyframe_image):

        if isinstance(keyframe_image, bytes): # image passed in base64 representation
            self.__image: np.ndarray = image_utils.base64_to_rgb(keyframe_image)
        elif isinstance(keyframe_image, np.ndarray): # image passed in numpy array representation
            self.__image = keyframe_image
        else:
            self.__image = None

        if(self.__image.all() != None):
            self.__json_image = json.dumps(self.__image, cls = numpy_encoder.NumpyEncoder)

        self.__face_detection_results = None
    
    def to_json(self):
        return json.dumps(self.__dict__, indent = 4)

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def json_image(self):
        return self.__json_image

    @property
    def face_detection_results(self):
        return self.__face_detection_results

    @face_detection_results.setter
    def face_detection_results(self, value):
        self.__face_detection_results = value
