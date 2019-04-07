from typing import List

from .image_face_detector import ImageFaceDetector
from .face_detection_results import FaceDetectionResults
from ..entities.image import KeyFrame

class VideoFaceDetector():

    __image_face_detector: ImageFaceDetector

    def __init__(self, image_face_detector: ImageFaceDetector):
        self.__image_face_detector = image_face_detector

    def get_detected_results(self, key_frames: List[KeyFrame]) -> List[FaceDetectionResults]:
        """Returns a List of Face Detection result corresponding to each received Key Frame"""
        return list(map(
            lambda key_frame: self.__image_face_detector.detect(key_frame.image),
                key_frames))