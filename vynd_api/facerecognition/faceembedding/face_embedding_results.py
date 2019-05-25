
from typing import NamedTuple, List

from .face_embedding import FaceEmbedding

class FaceEmbeddingResults(NamedTuple):
    """
    Result returned by the FaceEmbedder for a single KeyFrame, which may have more than one detected Face
    - faces: List[FaceEmbedding]
    """
    faces: List[FaceEmbedding]
