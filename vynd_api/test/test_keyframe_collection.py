from typing import List

from ..data.keyframe_collection import KeyFrameCollection
from ..entities.video import Video
from ..entities.image import KeyFrame
from .test_utils import get_img_from_filename

import numpy as np
import unittest

class TestKeyFrameCollection(unittest.TestCase):

    def setUp(self):
        img = get_img_from_filename('faced.jpg')
        self.keyframe = KeyFrame(img)
        self.keyframe_collection = KeyFrameCollection()

    def test_keyframe_collection(self):
        video_id = "dummy_id"
        keyframe_id = self.keyframe_collection.insert_keyframe(video_id = video_id, 
                                                               order_in_time = 1, 
                                                               key_frame = self.keyframe)
        self.assertEqual(len(keyframe_id), 24)                                                       

if __name__ == '__main__':
    unittest.main()