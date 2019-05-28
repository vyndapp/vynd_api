
from typing import List

import unittest
import numpy as np

from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facerecognition.faceembedding.face_embedding import FaceEmbedding
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..entities.keyframe import KeyFrame
from ..test.test_utils import get_img_from_filename

class TestVGGFaceRecognizer(unittest.TestCase):
    
    def setUp(self):
        self.vggface: ImageFacesEmbedder = VGGFaceEmbedder()
        self.faced: ImageFaceDetector = FacedDetector(minimum_confidence=0.9, 
                                                      offset_value=20,
                                                      pad_value=10)

        img_path: str = 'resources/faced.jpg'
        self.keyframe = KeyFrame(get_img_from_filename(img_path))
        self.keyframe.video_id = "dummy_video_id"
        self.keyframe.keyframe_id = "dummy_keyframe_id"

        self.epsilon: float = 0.0001
    
    def test_vggface_recognizer(self):
        detection_result: FaceDetectionResults = self.faced.detect(keyframe=self.keyframe)
        embedding_result: List[FaceEmbedding] = self.vggface.faces_to_embeddings(detection_results=detection_result)
        
        expected_number_of_faces = 5
        expected_number_of_features = 512
        expected_confidences = [0.9987294, 0.9990884, 0.9980178, 0.99325585, 0.9943262]
        
        self.assertEqual(len(embedding_result), expected_number_of_faces)    

        for i in range(len(embedding_result)):
            self.assertEqual(embedding_result[i].keyframe_id, self.keyframe.keyframe_id)
            self.assertEqual(embedding_result[i].video_id, self.keyframe.video_id)
            self.assertEqual(len(embedding_result[i].features), expected_number_of_features)
            self.assertAlmostEqual(embedding_result[i].confidence, expected_confidences[i], delta=self.epsilon)
