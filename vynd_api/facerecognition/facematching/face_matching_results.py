
from typing import NamedTuple, List

from .face_match import FaceMatch

class FaceMatchingResults(NamedTuple):
    """
    Result returned by the FaceMatching for the list of FaceEmbeddings
    - matched_faces: List of FaceMatch
    """
    matched_faces: List[FaceMatch]
