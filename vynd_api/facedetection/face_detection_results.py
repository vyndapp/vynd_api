"""The results returned by the Face Detection Module"""
from typing import List, Optional, NamedTuple

import numpy as np
from .bounding_box import BoundingBox
from .face_detection_status import FaceDetectionStatus

class DetectedFace(NamedTuple):

    """
        - bbox: bounding boxes of face in the original image
        - image: np.array of the cropped face
    """

    bbox: BoundingBox
    image: np.array

class FaceDetectionResults:
    """
        The results returned by the Face Detection Module per frame:
        - status: an enumerator for the status of the detection
        - detected_faces: a list of detected faces in the processed frame
    """
    
    def __init__(self, status: FaceDetectionStatus, detected_faces: List[DetectedFace] = None):
        self.__detected_faces: Optional[List[DetectedFace]] = detected_faces
        self.__status: FaceDetectionStatus = status
    
    @property
    def detected_faces(self) -> Optional[List[DetectedFace]]:
        return self.__detected_faces

    @property
    def status(self) -> FaceDetectionStatus:
        return self.__status
