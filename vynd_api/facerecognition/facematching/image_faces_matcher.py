
from typing import List, Dict, Collection

import numpy as np
import json

from . import CLIENT
from . import FaceCollection
from . import FaceEmbedding
from .face_matching_results import FaceMatchingResults
from .face_match_status import FaceMatchStatus
from .face_match import FaceMatch
from . import recognition_utils
from . import numpy_encoder
from . import db_utils

class ImageFacesMatcher():

    def __init__(self, face_collection: Collection=CLIENT.vynd_db_test.face_collection):
        self.__face_collection = FaceCollection(face_collection)
        self.__similarity_threshold = 0.3

    def match_faces(self, face_embeddings: List[FaceEmbedding]) -> FaceMatchingResults:
        """
        - Matches FaceEmbeddings for a specific KeyFrame with Faces previously stored in DB
        - If there are faces that are not matched, they are inserted to the DB
        - Returns: FaceMatchingResults
        """
        self.__all_faces = self.__face_collection.get_all_faces()

        matched_faces: List[FaceMatch] = self.__find_most_similar_face(face_embeddings)
        self.__update_db(matched_faces, face_embeddings)

        return FaceMatchingResults(matched_faces=matched_faces)

    def __update_db(self, face_matches: List[FaceMatch], face_embeddings: List[FaceEmbedding]) -> None:
        """
        - Update previously added faces with new features/keyframes/videos
        - Insert new faces to DB
        """
        for (face_match, face_embedding) in zip(face_matches, face_embeddings):
            if(face_match.face_match_status == FaceMatchStatus.UNKNOWN_FACE):
                face_id = self.__face_collection.insert_new_face(keyframe_id=face_embedding.keyframe_id,
                                                                 video_id=face_embedding.video_id,
                                                                 features=face_embedding.features,
                                                                 face_image=face_embedding.face_image,
                                                                 confidence=face_embedding.confidence)
                face_match.face_id = face_id                                                                 
            else:
                face = self.__face_collection.get_face_by_id(face_match.most_similar_face_id)
                # print('face already existed, updating...')
                r4 = self.__face_collection.add_keyframe_id(face_match.most_similar_face_id, face_embedding.keyframe_id)
                r5 = self.__face_collection.add_video_id(face_match.most_similar_face_id, face_embedding.video_id)
                if(face_embedding.confidence > face['confidence_score']):
                    # print('update features, image, confidence', end=' ')
                    r1 = self.__face_collection.update_features(face_match.most_similar_face_id, face_embedding.features)
                    r2 = self.__face_collection.update_face_image(face_match.most_similar_face_id, face_embedding.face_image)
                    r3 = self.__face_collection.update_confidence_score(face_match.most_similar_face_id, face_embedding.confidence)
                    # print(r1, r2, r3, r4, r5)
                face_match.face_id = face_match.most_similar_face_id

    def __find_most_similar_face(self, face_embeddings: List[FaceEmbedding]) -> List[FaceMatch]:
        face_matches = []

        for face_embedding in face_embeddings:
            embedding1 = face_embedding.features
            min_cosine_similarity_distance = np.inf
            most_similar_id = ""
            most_similar_name = ""

            for face in self.__all_faces.rewind():
                embedding2 = db_utils.binary_to_np(face['features'])
                cosine_similarity_dist = recognition_utils.cosine_similarity_distance(embedding1, 
                                                                                      embedding2)

                if(cosine_similarity_dist < min_cosine_similarity_distance):
                    min_cosine_similarity_distance = cosine_similarity_dist
                    most_similar_id = str(face['_id'])
                    most_similar_name = face['name']

            # print(face_embedding.keyframe_id, most_similar_id, min_cosine_similarity_distance)
            if(min_cosine_similarity_distance < self.__similarity_threshold):
                face_matches.append(FaceMatch(features=embedding1,
                                              most_similar_face_id=most_similar_id,
                                              face_match_status=FaceMatchStatus.MATCHED,
                                              most_similar_face_name=most_similar_name))
            else:
                face_matches.append(FaceMatch(features=embedding1,
                                              most_similar_face_id=most_similar_id,
                                              face_match_status=FaceMatchStatus.UNKNOWN_FACE,
                                              most_similar_face_name=most_similar_name))
        
        return face_matches
