
import unittest
import numpy as np
import json

from ..utils import numpy_encoder
from ..data import CLIENT
from ..data.face_collection import FaceCollection

class TestFaceCollection(unittest.TestCase):

   def setUp(self):
      self.face_collection = FaceCollection(collection=CLIENT.vynd_db_test.face_collection)

   def test_insert_face(self):
      features = np.random.uniform(low=0.5, high=10, size=(5,))
      self.face_collection.insert_face(
         features=json.dumps(features, cls=numpy_encoder.NumpyEncoder),
         cosine_similarity_distance=0.3,
         euclidean_distance=0.7
      )
      

