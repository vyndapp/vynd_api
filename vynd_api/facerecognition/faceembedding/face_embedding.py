
from typing import NamedTuple

import numpy as np

class FaceEmbedding(NamedTuple):
    """
    The feature vector that represents the DetectedFace:
    - video_id: str
    - keyframe_id: str
    - feature: np.ndarray
    - face_image: np.ndarray
    """
    video_id: str
    keyframe_id: str
    features: np.ndarray
    face_image: np.ndarray
