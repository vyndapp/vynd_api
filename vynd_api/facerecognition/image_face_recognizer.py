
from abc import ABC, abstractmethod
from typing import List, Optional, NamedTuple

from ..data.face_collection import FaceCollection
from .face_recognition_results import FaceRecognitionResults
from ..facedetection.face_detection_results import DetectedFace

import numpy as np

class ImageFaceRecognizer(ABC):

    # FaceCollection: self._face_collection
    def __init__(self, face_collection: FaceCollection):
        self._face_collection = face_collection

    @abstractmethod
    def recognize(self, detected_faces: List[DetectedFace]) -> FaceRecognitionResults:
        pass
        
        