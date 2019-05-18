
from typing import NamedTuple

import numpy as np

class FaceEmbedding(NamedTuple):
    """
    The feature vector that represents the DetectedFace:
    - feature: np.ndarray
    - face_image: np.ndarray
    - confidence: float
    """
    features: np.ndarray
    face_image: np.ndarray
    confidence: float