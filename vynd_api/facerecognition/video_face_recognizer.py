
from typing import List, Collection, Tuple

from . import CLIENT
from .faceembedding.image_faces_embedder import ImageFacesEmbedder
from .facematching.image_faces_matcher import ImageFacesMatcher
from .facematching.face_match import FaceMatch
from .facematching.face_matching_results import FaceMatchingResults
from .faceembedding.face_embedding_results import FaceEmbeddingResults
from . import FaceDetectionResults

class VideoFaceRecognizer():
    __image_faces_embedder: ImageFacesEmbedder
    __image_faces_matcher: ImageFacesMatcher

    def __init__(self, image_faces_embedder: ImageFacesEmbedder, face_collection: Collection=CLIENT.vynd_db_test.face_collection):
        self.__image_faces_embedder = image_faces_embedder
        self.__image_faces_matcher: ImageFacesMatcher = ImageFacesMatcher(face_collection)

    def recognize(self, face_detection_results: List[FaceDetectionResults]) -> List[FaceMatch]:
        embedding_results = list(map(
                                lambda detection_result: self.__image_faces_embedder.faces_to_embeddings(detection_result),
                                face_detection_results))
        
        matching_results = list(map(lambda embedding_result: self.__image_faces_matcher.match_faces(embedding_result), embedding_results))
        return matching_results    
