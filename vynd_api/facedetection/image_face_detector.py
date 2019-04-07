from abc import ABC, abstractmethod
from .face_detection_results import FaceDetectionResults

class ImageFaceDetector(ABC):
    
    def __init__(self, minimum_confidence = 0.8, offset_value = 20):
        self.__minimum_confidence = minimum_confidence
        self.__offset_value = offset_value
    
    @abstractmethod
    def detect(self, image) -> FaceDetectionResults:
        pass