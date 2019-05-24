
from typing import NamedTuple, List

from .face_embedding import FaceEmbedding

class FaceEmbeddingResults(NamedTuple):
    """
    Result returent by the FaceEmbedder for a single KeyFrame, which may have more thanone detected Face
    - faces: List[FaceEmbedding]
    """
    faces: List[FaceEmbedding]
