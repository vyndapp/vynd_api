
import unittest
import numpy as np

from ..facerecognition.vggface2 import VGGFaceRecognizer

class TestVGGFaceRecognizer(unittest.TestCase):
    
    def setUp(self):
        self.vggface = VGGFaceRecognizer()
    
    def test_vggface(self):
        img = np.random.rand(256, 256, 3)
        embedding = self.vggface.recognize(img).tolist()
        self.assertEqual(len(embedding), 512)