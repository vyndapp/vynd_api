from typing import List

from .faceembedding.image_faces_embedder import ImageFacesEmbedder
from .faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facedetection.face_detection_results import FaceDetectionResults

class VideoFaceRecognizer():
    __image_faces_embedder: ImageFacesEmbedder

    def __init__(self, image_faces_embedder: ImageFacesEmbedder):
        self.__image_faces_embedder = image_faces_embedder

    def recognize(self, face_detection_results: List[FaceDetectionResults]) -> List[FaceEmbeddingResults]:
        """Returns a List of Face Recognition result corresponding to each received Face Detection Result"""
        return list(map(
            lambda detection_results: self.__image_faces_embedder.faces_to_embeddings(detection_results.detected_faces),
            face_detection_results))
