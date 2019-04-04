from abc import ABC, abstractmethod
from .facedetectionresults import FaceDetectionResults

class ImageFaceDetector(ABC):
    @abstractmethod
    def detect(self, image) -> FaceDetectionResults:
        pass

