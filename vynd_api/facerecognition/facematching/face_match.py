
from typing import NamedTuple

import numpy as np

from .face_match_status import FaceMatchStatus

class FaceMatch(NamedTuple):
    most_similar_face_id: str
    face_match_status: FaceMatchStatus