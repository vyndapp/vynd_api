
from typing import List

import unittest
import numpy as np

from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facerecognition.faceembedding.face_embedding_status import FaceEmbeddingStatus
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..entities.keyframe import KeyFrame
from ..test.test_utils import get_img_from_filename

class TestVGGFaceRecognizer(unittest.TestCase):
    
    def setUp(self):
        self.vggface: ImageFacesEmbedder = VGGFaceEmbedder()
        self.faced: ImageFaceDetector = FacedDetector(minimum_confidence=0.8, offset_value=20)

        img_path: str = 'resources/faced.jpg'
        self.keyframe = get_img_from_filename(img_path)

        self.epsilon: float = 0.0001
    
    def test_vggface_recognizer(self):
        detection_result: FaceDetectionResults = self.faced.detect(image=self.keyframe)
        recognition_result: FaceEmbeddingResults = self.vggface.faces_to_embeddings(detected_faces=detection_result.detected_faces)
        
        expected_number_of_faces = 5
        expected_number_of_features = 512
        expected_confidences = [0.99901056, 0.99876785, 0.99876535, 0.9876309, 0.9892281]
        expected_means = [0.001628648, 0.002705003, 0.0019555842, -2.4607521e-05, -0.00025454361]
        expected_status: FaceEmbeddingStatus = FaceEmbeddingStatus.SUCCESS
        
        self.assertEqual(recognition_result.status, expected_status)
        self.assertEqual(len(recognition_result.faces), expected_number_of_faces)    

        for i in range(len(recognition_result.faces)):
            self.assertEqual(len(recognition_result.faces[i].features), expected_number_of_features)
            self.assertAlmostEqual(np.mean(recognition_result.faces[i].features), expected_means[i], delta=self.epsilon)
            self.assertAlmostEqual(recognition_result.faces[i].confidence, expected_confidences[i], delta=self.epsilon)
