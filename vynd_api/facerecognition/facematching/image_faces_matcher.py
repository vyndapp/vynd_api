from typing import List, Optional
# from dataclasses import dataclass

import numpy as np

from . import CLIENT
from . import FaceCollection
from . import FaceEmbedding
from .face_match_status import FaceMatchStatus
from ...utils.recognition_utils import cosine_similarity_distance
from ..facegrouping.face_grouping import group_faces

# @dataclass(init=True)
class GroupMatch():
    face_embeddings: List[FaceEmbedding]
    match_status: FaceMatchStatus
    matched_id: Optional[str]
    
    def __init__(self, face_embeddings: List[FaceEmbedding], match_status: FaceMatchStatus, matched_id: Optional[str]):
        self.face_embeddings = face_embeddings
        self.match_status = match_status
        self.matched_id = matched_id

class ImageFacesMatcher():

    def __init__(self, face_collection=CLIENT.vynd_db.face_collection):
        self.__face_collection = FaceCollection(face_collection)
        self.__similarity_distance_threshold = 0.3
        self.__default_face_dims = (100, 100)

    def match_faces(self, face_embeddings: List[FaceEmbedding]) -> List[GroupMatch]:
        """
        - Matches FaceEmbeddings for a specific all KeyFrame with Faces previously stored in DB
        - If there are faces that are not matched, they are inserted to the DB
        - Returns: FaceMatchingResults
        """
        all_faces = self.__face_collection.get_all_faces_features()
        face_groups: List[List[FaceEmbedding]] = group_faces(face_embeddings, \
         lambda features_a, features_b: cosine_similarity_distance(features_a, features_b) < 0.28)

        group_matches: List[GroupMatch] = self.__get_group_matches(face_groups, all_faces)

        return group_matches

    def __get_group_matches(self, groups: List[List[FaceEmbedding]], all_faces) -> List[GroupMatch]:

        def find_most_similar_face(group: List[FaceEmbedding]):
            best_sum, best_face = np.inf, None
            for face in all_faces:
                face_sum = 0
                for embedding in group:
                    distance = cosine_similarity_distance(embedding.features, face['features'])
                    if distance == 0.0:
                        return face['_id']
                    elif distance < self.__similarity_distance_threshold:
                        face_sum += distance
                if (face_sum != 0 and face_sum < best_sum):
                    best_sum, best_face = face_sum, face['_id']
            return best_face

        def get_group_match(group) -> GroupMatch:
            face_id = find_most_similar_face(group)
            match_status = FaceMatchStatus.MATCHED if face_id else FaceMatchStatus.UNKNOWN_FACE
            return GroupMatch(face_embeddings=group, match_status=match_status, matched_id=face_id)

        return list(map(get_group_match, groups))
        