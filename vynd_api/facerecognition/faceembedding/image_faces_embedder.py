
from abc import ABC, abstractmethod
from typing import List

from . import FaceCollection
from .face_embedding import FaceEmbedding
from .. import FaceDetectionResults
from . import DetectedFace

class ImageFacesEmbedder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def faces_to_embeddings(self, detection_results: FaceDetectionResults) -> List[FaceEmbedding]:
        pass