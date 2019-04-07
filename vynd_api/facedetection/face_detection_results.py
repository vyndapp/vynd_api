"""The results returned by the Face Detection Module"""
from typing import List, Optional

from .bounding_box import BoundingBox
from .face_detection_status import FaceDetectionStatus

class FaceDetectionResults:
    """
        The results returned by the Face Detection Module:
        - status: an enumerator for the status of the detection
        - bboxes: a list of bounding boxes related to the detection
    """
    
    def __init__(self, status: FaceDetectionStatus, bboxes: List[BoundingBox] = None):
        self.__bboxes: Optional[List[BoundingBox]] = bboxes
        self.__status: FaceDetectionStatus = status
    
    @property
    def bboxes(self) -> Optional[List[BoundingBox]]:
        return self.__bboxes
    
    @property
    def status(self) -> FaceDetectionStatus:
        return self.__status
