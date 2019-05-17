
from typing import List, Dict, Collection

import numpy as np
import json

from . import CLIENT
from . import FaceCollection
from . import FaceEmbedding
from .face_matching_results import FaceMatchingResults
from .face_matching_status import FaceMatchingStatus
from .face_match import FaceMatch
from . import recognition_utils
from . import numpy_encoder

class ImageFacesMatcher():

    def __init__(self, face_collection: Collection=CLIENT.vynd_db_test.face_collection):
        self.__face_collection = FaceCollection(face_collection)

    def match_faces(self, face_embeddings: List[FaceEmbedding]) -> FaceMatchingResults:
        self.__all_faces = self.__face_collection.get_all_faces()

        face_matches = self.__find_most_similar_face(face_embeddings)
        self.__update_db(face_matches)
                               
        return FaceMatchingResults(matching_faces=face_matches,
                                   status=FaceMatchingStatus.MATCHED)

    def __update_db(self, face_matches: List[FaceMatch]) -> None:
        for face_match in face_matches:
            if(face_match.matched_face_id == "" or face_match.cosine_similarity_distance > 0.3):
                self.__face_collection.insert_face(features=face_match.features,
                                                   cosine_similarity_distance=min(10, face_match.cosine_similarity_distance))
            else:
                matched_face = self.__face_collection.get_face_by_id(face_match.matched_face_id)
                if(face_match.cosine_similarity_distance < matched_face.get("cosine_similarity_distance")):
                    self.__face_collection.update_distance(face_match.matched_face_id, 
                                                           face_match.cosine_similarity_distance)

    def __find_most_similar_face(self, face_embeddings: List[FaceEmbedding]) -> List[FaceMatch]:
        face_matches = []

        for face_embedding in face_embeddings:
            embedding1 = face_embedding.features
            min_cosine_similarity_distance = np.inf
            most_similar_id = ""

            for face in self.__all_faces.rewind():
                cosine_similarity_dist = recognition_utils.cosine_similarity_distance(embedding1, 
                                                                                      np.array(json.loads(face["features"])))

                if(cosine_similarity_dist <= min_cosine_similarity_distance):
                    min_cosine_similarity_distance = cosine_similarity_dist
                    most_similar_id = face["_id"]
                    
            face_matches.append(FaceMatch(features=json.dumps(embedding1, cls=numpy_encoder.NumpyEncoder),
                                          cosine_similarity_distance=min_cosine_similarity_distance,
                                          matched_face_id=most_similar_id))
        
        return face_matches
