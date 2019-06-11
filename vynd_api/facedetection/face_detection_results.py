"""The results returned by the Face Detection Module"""
from typing import List, NamedTuple

import numpy as np

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
        - status: an enumerator for the status of the detection
        - detected_faces: a list of detected faces in the processed frame
    """
    def __init__(self, status: FaceDetectionStatus, detected_faces: List[DetectedFace] = None):
        self.__detected_faces: List[DetectedFace] = detected_faces
        self.__status: FaceDetectionStatus = status
    
    @property
    def detected_faces(self) -> List[DetectedFace]:
        return self.__detected_faces

    @property
    def status(self) -> FaceDetectionStatus:
        return self.__status