from abc import ABC, abstractmethod

from ..data.face_collection import FaceCollection

class ImageFaceRecognizer(ABC):

    # FaceCollection: self._face_collection
    def __init__(self, face_collection: FaceCollection):
        self._face_collection = face_collection

    @abstractmethod
    def recognize(self, image):
        pass
        