
from typing import List

import unittest
import numpy as np

from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults
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
        self.keyframe = get_img_from_filename(img_path)

        self.epsilon: float = 0.0001
    
    def test_vggface_recognizer(self):
        detection_result: FaceDetectionResults = self.faced.detect(image=self.keyframe)
        with self.vggface:
            embedding_results: FaceEmbeddingResults = self.vggface.faces_to_embeddings(detected_faces=detection_result.detected_faces)
        
        expected_number_of_faces = 5
        expected_number_of_features = 512
        expected_confidences = [0.9987294, 0.9990884, 0.9980178, 0.99325585, 0.9943262]
        expected_means = [0.0011961971, 0.001964572, -0.00022213103, -0.00057893054, 0.0003350584]
        
        self.assertEqual(len(embedding_results.faces), expected_number_of_faces)    

        for i in range(len(embedding_results.faces)):
            self.assertEqual(len(embedding_results.faces[i].features), expected_number_of_features)
            self.assertAlmostEqual(np.mean(embedding_results.faces[i].features), expected_means[i], delta=self.epsilon)
            self.assertAlmostEqual(embedding_results.faces[i].confidence, expected_confidences[i], delta=self.epsilon)
