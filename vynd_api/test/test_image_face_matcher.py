
from typing import List

import unittest
import numpy as np

from ..data.face_collection import FaceCollection
from ..data import CLIENT


from ..facerecognition.facematching.face_match_status import FaceMatchStatus
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.face_embedding import FaceEmbedding
from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher, GroupMatch
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..entities.keyframe import KeyFrame
from ..test.test_utils import get_img_from_filename

class TestImageFaceMatcher(unittest.TestCase):
    
    def setUp(self):
        self.embedder: ImageFacesEmbedder = VGGFaceEmbedder()
        self.faced: ImageFaceDetector = FacedDetector(minimum_confidence=0.9, 
                                                      offset_value=20,
                                                      pad_value=10)
        self.matcher: ImageFacesMatcher = ImageFacesMatcher(face_collection=CLIENT.vynd_db_test.face_collection)

        self.face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)                                        

        img_path: str = 'resources/1.jpg' ## group photo of 4 persons
        self.keyframe = KeyFrame(get_img_from_filename(img_path))
        self.keyframe.video_id = "dummy_video_id"
        self.keyframe.keyframe_id = "dummy_keyframe_id"

        self.epsilon: float = 0.0001

    def tearDown(self):
        self.face_collection.delete_all_faces()
    
    def insert_new_face_in_db(self, img_path):
        img = get_img_from_filename(img_path)
        embedding = self.embedder.image_to_embedding(img)
        return self.face_collection.insert_new_face(video_id="123", \
                                                    features=embedding,
                                                    face_image=None,
                                                    )
    def test_image_face_matcher(self):

        face_id_in_db_1 = self.insert_new_face_in_db('resources/zuma.jpg')
        self.insert_new_face_in_db('resources/bahi.jpg')

        detection_result: FaceDetectionResults = self.faced.detect(keyframe=self.keyframe)
        embedding_results: List[FaceEmbedding] = self.embedder.faces_to_embeddings(detection_results=detection_result)
        group_matches: List[GroupMatch] = self.matcher.match_faces(embedding_results)

        self.assertEqual(len(group_matches), 4)

        match_found = 0
        for group_match in group_matches:
            if group_match.match_status == FaceMatchStatus.MATCHED:
                match_found += 1
                self.assertEqual(face_id_in_db_1, group_match.matched_id)
        self.assertEqual(match_found, 1)