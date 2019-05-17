
from typing import NamedTuple, List

from .face_embedding import FaceEmbedding
from .face_embedding_status import FaceEmbeddingStatus

class FaceEmbeddingResults(NamedTuple):
    """
    Result returned by the Face Recognition for a single Key Frame, which may have
    more than one cropped Face. A FaceResult for each such image
    - status : Enumerator that signals the status of recognition
    - faces: List of found faces in the processed frame
    """
    faces: List[FaceEmbedding]
    status: FaceEmbeddingStatus
