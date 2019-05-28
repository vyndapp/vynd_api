
from typing import List

import unittest
import numpy as np

from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.face_embedding import FaceEmbedding
from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..entities.keyframe import KeyFrame
from ..test.test_utils import get_img_from_filename

class TestImageFaceMatcher(unittest.TestCase):
    
    def setUp(self):
        self.vggface: ImageFacesEmbedder = VGGFaceEmbedder()
        self.faced: ImageFaceDetector = FacedDetector(minimum_confidence=0.9, 
                                                      offset_value=20,
                                                      pad_value=10)
        self.matcher: ImageFacesMatcher = ImageFacesMatcher()                                                 

        img_path: str = 'resources/zuma.jpg'
        self.keyframe = KeyFrame(get_img_from_filename(img_path))
        self.keyframe.video_id = "dummy_video_id"
        self.keyframe.keyframe_id = "dummy_keyframe_id"

        self.epsilon: float = 0.0001
    
    def test_image_face_matcher(self):
        detection_result: FaceDetectionResults = self.faced.detect(keyframe=self.keyframe)
        embedding_results: List[FaceEmbedding] = self.vggface.faces_to_embeddings(detection_results=detection_result)
        
        names = ['hazem']

        matching_results = self.matcher.match_faces(embedding_results)

        self.assertEqual(len(names), len(matching_results.matched_faces))
        
        for (face, name) in zip(matching_results.matched_faces, names):
            self.assertEqual(face.most_similar_face_name, name)