
from typing import NamedTuple

import numpy as np

from .face_match_status import FaceMatchStatus

class FaceMatch:
    """
    FaceMatch:
    - face_id: str
    - feature: np.ndarray
    - most_similar_face_id: str
    - most_similar_face_name: str
    - face_match_status: FaceMatchStatus
    """
    __face_id: str

    def __init__(self, features: np.ndarray, most_similar_face_id: str, most_similar_face_name: str, face_match_status: FaceMatchStatus):
        self.__features = features
        self.__most_similar_face_id = most_similar_face_id
        self.__most_similar_face_name = most_similar_face_name
        self.__face_match_status = face_match_status
        
    @property
    def most_similar_face_id(self):
        return self.__most_similar_face_id
    
    @property
    def face_match_status(self):
        return self.__face_match_status

    @property
    def most_similar_face_name(self):
        return self.__most_similar_face_name
    
    @property
    def face_id(self):
        return self.__face_id

    @face_id.setter
    def face_id(self, value):
        self.__face_id = value