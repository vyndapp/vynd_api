

import unittest
import numpy as np

from ..facerecognition.facenet import FaceNetRecognizer

class TestFaceNetRecognizer(unittest.TestCase):
    
    def setUp(self):
        self.facenet = FaceNetRecognizer()
    
    def test_facenet(self):
        img = np.random.rand(256, 256, 3)
        embedding = self.facenet.recognize(img).tolist()
        self.assertEqual(len(embedding), 128)