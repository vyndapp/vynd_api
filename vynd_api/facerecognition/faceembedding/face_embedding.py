
from typing import NamedTuple

import numpy as np

class FaceEmbedding(NamedTuple):
    features: np.ndarray
    confidence: float