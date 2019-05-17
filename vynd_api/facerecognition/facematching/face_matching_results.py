
from typing import NamedTuple, List

from .face_match import FaceMatch
from .face_matching_status import FaceMatchingStatus

class FaceMatchingResults(NamedTuple):
    """
    Result returned by the Face Recognition for a single Key Frame, which may have
    more than one cropped Face. A FaceResult for each such image
    - status : Enumerator that signals the status of recognition
    - faces: List of found faces in the processed frame
    """
    matching_faces: List[FaceMatch]
    status: FaceMatchingStatus
