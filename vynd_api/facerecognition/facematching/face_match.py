
from typing import NamedTuple

import numpy as np

class FaceMatch(NamedTuple):
    features: np.ndarray
    cosine_similarity_distance: float
    matched_face_id: str