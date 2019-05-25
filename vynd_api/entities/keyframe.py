"""Canonical representations for images being handled"""

from typing import List, Optional, Set

import numpy as np
import json

from ..utils import image_utils, numpy_encoder
from ..facedetection.face_detection_results import FaceDetectionResults

class KeyFrame:
    """
    KeyFrame:
    - keyframe_id: str
    - video_id: str
    - faces_ids: Set[str]
    - timestamp: Optional[int]
    - image: np.ndarray
    - json_image: str
    """
    __keyframe_id: str
    __video_id: str
    __faces_ids: Set[str]
    __timestamp: Optional[int]

# TODO: video_id must be a mandatory property
    def __init__(self, keyframe_image, timestamp: Optional[int]=0):
        """
        Params:
        - keyframe_image: numpy.ndarray | bytes
        - timestamps: Optional[int]=0
        """
        self.__timestamp = timestamp
        self.__image = self.__process_keyframe_image(keyframe_image)
        self.__json_image = self.__save_json_image(self.__image)
        self.__faces_ids = set()

    def __save_json_image(self, keyframe_image):
        if(keyframe_image.all() != None):
            return json.dumps(keyframe_image, cls = numpy_encoder.NumpyEncoder) 
        else:
            return None
    
    def __process_keyframe_image(self, keyframe_image):
        if isinstance(keyframe_image, bytes): # image passed in base64 representation
            return image_utils.base64_to_rgb(keyframe_image)
        elif isinstance(keyframe_image, np.ndarray): # image passed in numpy array representation
            return keyframe_image
        else:
            return None

    def add_face(self, face_id: str)-> None:
        self.__faces_ids.add(face_id)

    @property
    def keyframe_id(self):
        return self.__keyframe_id

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def json_image(self):
        return self.__json_image

    @property
    def video_id(self):
        return self.__video_id

    @property
    def faces_ids(self):
        return self.__faces_ids

    @property
    def timestamp(self):
        return self.__timestamp

    @keyframe_id.setter
    def keyframe_id(self, value):
        self.__keyframe_id = value

    @video_id.setter
    def video_id(self, value):
        self.__video_id = value
