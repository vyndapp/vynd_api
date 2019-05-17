
from abc import ABC, abstractmethod
from typing import List

from . import FaceCollection
from .face_embedding_results import FaceEmbeddingResults
from . import DetectedFace

class ImageFacesEmbedder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def faces_to_embeddings(self, detected_faces: List[DetectedFace]) -> FaceEmbeddingResults:
        pass
        
        