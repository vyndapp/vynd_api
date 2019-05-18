from enum import Enum

class FaceDetectionStatus(Enum):
    """
    Enumerators that represent the status of the face detection result:
    - SUCCESS
    - FAIL_NON_RGB_INPUT: when the image does not contain extacly 3 channels (RGB)
    """
    SUCCESS = 'success'
    FAIL_NON_RGB_INPUT = 'image does not contain 3 channels'