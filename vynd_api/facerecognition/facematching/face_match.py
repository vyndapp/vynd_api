
from typing import NamedTuple

import numpy as np

from .face_match_status import FaceMatchStatus

class FaceMatch(NamedTuple):
    """
    FaceMatch:
    - feature: np.ndarray
    - most_similar_face_id: str
    - face_match_status: FaceMatchStatus
    """
    features: np.ndarray
    most_similar_face_id: str
    most_similar_face_name: str
    face_match_status: FaceMatchStatus