"""Canonical representations for images being handled"""

from ..utils import imageutils
from ..facedetection.facedetectionresults import FaceDetectionResults

# frame timestamp can be added later as an attribute
class KeyFrame:
    
    _face_detection_results: FaceDetectionResults

    def __init__(self, base_64_image_string):
        self._image = imageutils.base64_to_rgb(base_64_image_string)
        self._face_detection_results = None

    @property
    def image(self):
        return self._image

    @property
    def face_detection_results(self):
        return self._face_detection_results

    @face_detection_results.setter
    def face_detection_results(self, value):
        self._face_detection_results = value
