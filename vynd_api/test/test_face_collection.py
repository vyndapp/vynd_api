
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
      features = np.random.uniform(low=0.5, high=10, size=(512,))
      face = np.random.uniform(low=0.5, high=10, size=(10, 10))
      self.face_collection.insert_new_face(
         keyframe_id="dummy_keyframe_id",
         video_id="dummy_video_id",
         features=json.dumps(features, cls=numpy_encoder.NumpyEncoder),
         face_image=face,
         confidence=0.9
      )
      

