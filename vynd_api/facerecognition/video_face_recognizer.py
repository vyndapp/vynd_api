
from typing import List, Collection

from ..data import CLIENT
from .faceembedding.image_faces_embedder import ImageFacesEmbedder
from .facematching.image_faces_matcher import ImageFacesMatcher
from .facematching.face_matching_results import FaceMatchingResults
from .faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facedetection.face_detection_results import FaceDetectionResults

class VideoFaceRecognizer():
    __image_faces_embedder: ImageFacesEmbedder
    __image_faces_matcher: ImageFacesMatcher

    def __init__(self, image_faces_embedder: ImageFacesEmbedder, face_collection: Collection=CLIENT.vynd_db_test.face_collection):
        self.__image_faces_embedder = image_faces_embedder
        self.__image_faces_matcher: ImageFacesMatcher = ImageFacesMatcher(face_collection)

    def recognize(self, face_detection_results: List[FaceDetectionResults]) -> List[FaceEmbeddingResults]:
        with self.__image_faces_embedder:
            embedding_results: List[FaceEmbeddingResults] = list(map(
                                                                     lambda detection_result: self.__image_faces_embedder.faces_to_embeddings(detection_result.detected_faces),
                                                                     face_detection_results))

        matching_results: List[FaceMatchingResults] = list(map(
                                                               lambda embedding_result: self.__image_faces_matcher.match_faces(embedding_result.faces),
                                                               embedding_results))
        
        return embedding_results, matching_results
