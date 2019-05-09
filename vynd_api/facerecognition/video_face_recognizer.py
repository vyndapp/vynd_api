from typing import List

from .image_face_recognizer import ImageFaceRecognizer
from .face_recognition_results import FaceRecognitionResults
from ..facedetection.face_detection_results import FaceDetectionResults

class VideoFaceRecognizer():
    __image_face_recognizer: ImageFaceRecognizer

    def __init__(self, image_face_recognizer: ImageFaceRecognizer):
        self.__image_face_recognizer = image_face_recognizer

    def recognize(self, face_detection_results: List[FaceDetectionResults]) -> List[FaceRecognitionResults]:
        """Returns a List of Face Recognition result corresponding to each received Face Detection Result"""
        return list(map(
            lambda detection_results: self.__image_face_recognizer.recognize(detection_results.detected_faces),
            face_detection_results))
