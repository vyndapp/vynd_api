
from enum import Enum

class FaceMatchStatus(Enum):
    """
    - MATCHED
    - UNKNOWN_FACE
    """
    MATCHED = 'matched'
    UNKNOWN_FACE = 'unknown_face'