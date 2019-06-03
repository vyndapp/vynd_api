"""The results returned by the Face Detection Module"""
from typing import List, Optional, NamedTuple

import numpy as np
import json

from ..utils import numpy_encoder
from .bounding_box import BoundingBox
from .face_detection_status import FaceDetectionStatus

class DetectedFace(NamedTuple):

    """
        - image: np.array of the cropped face
        - aligned_image: np.array of aligned cropped face
    """

    image: np.array
    aligned_image: np.array

class FaceDetectionResults:
    """
        The results returned by the Face Detection Module per frame:
        - video_id: str
        - keyframe_id: str
        - status: an enumerator for the status of the detection
        - detected_faces: a list of detected faces in the processed frame
    """
    
    def __init__(self, video_id: str, keyframe_id: str, status: FaceDetectionStatus, detected_faces: List[DetectedFace] = None):
        self.__video_id = video_id
        self.__keyframe_id = keyframe_id
        self.__detected_faces: List[DetectedFace] = detected_faces
        self.__status: FaceDetectionStatus = status
    
    @property
    def detected_faces(self) -> List[DetectedFace]:
        return self.__detected_faces

    @property
    def status(self) -> FaceDetectionStatus:
        return self.__status

    @property
    def keyframe_id(self) -> str:
        return self.__keyframe_id

    @property
    def video_id(self) -> str:
        return self.__video_id
