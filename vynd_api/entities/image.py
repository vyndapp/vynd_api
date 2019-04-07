"""Canonical representations for images being handled"""

import numpy as np

from ..utils import image_utils
from ..facedetection.face_detection_results import FaceDetectionResults

# frame timestamp can be added later as an attribute
class KeyFrame:
    
    __face_detection_results: FaceDetectionResults

    def __init__(self, base_64_image_string):
        self.__image: np.ndarray = image_utils.base64_to_rgb(base_64_image_string)
        self.__face_detection_results = None

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def face_detection_results(self):
        return self.__face_detection_results

    @face_detection_results.setter
    def face_detection_results(self, value):
        self.__face_detection_results = value
