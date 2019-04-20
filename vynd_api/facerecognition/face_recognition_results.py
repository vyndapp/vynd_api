from typing import Optional, NamedTuple, List
from enum import Enum

import numpy as np

class FaceResult(NamedTuple):
    face_id: Optional[str]
    features: np.array

class FaceRecognitionStatus(Enum):
    SUCCESS = 'success'

class FaceRecognitionResults(NamedTuple):

    """
    Result returned by the Face Recognition for a single Key Frame, which may have
    more than one cropped Face. A FaceResult for each such image
    - status : Enumerator that signals the status of recognition
    - faces: List of found faces in the processed frame
    """
    faces: List[FaceResult]
    status: FaceRecognitionStatus
