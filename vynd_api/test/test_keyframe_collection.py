
from typing import List
from pymongo.results import DeleteResult

from ..data import CLIENT
from ..data.video_collection import VideoCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..entities.video import Video
from ..entities.keyframe import KeyFrame
from .test_utils import get_img_from_filename

import numpy as np
import unittest

class TestKeyFrameCollection(unittest.TestCase):

    def setUp(self):
        self.video_collection = VideoCollection(collection=CLIENT.vynd_db_test.video_collection)
        self.keyframe_collection = KeyFrameCollection(collection=CLIENT.vynd_db_test.keyframe_collection)

    def test_keyframe_collection(self):
        video_id = self.video_collection.insert_new_video()

        keyframe_id = self.keyframe_collection.insert_new_keyframe(video_id=video_id)       

        self.assertTrue(self.video_collection.add_keyframe(video_id, keyframe_id))

        faces = ["f1", "f2", "f3", "f4", "f1", "f2"]

        for f_id in faces:
            result = self.keyframe_collection.add_face(keyframe_id=keyframe_id, 
                                                       face_id=f_id)
            self.assertTrue(result)
        
        expected_number_of_faces = 4

        keyframe = self.keyframe_collection.get_keyframe_by_id(keyframe_id=keyframe_id)

        self.assertEqual(len(keyframe["faces_ids"]), expected_number_of_faces)

        keyframe_delete_result: DeleteResult = self.keyframe_collection.delete_keyframe(keyframe_id=keyframe_id)
        self.assertEqual(keyframe_delete_result.deleted_count, 1)

        video_delete_result: DeleteResult = self.video_collection.delete_video(video_id=video_id)
        self.assertEqual(video_delete_result.deleted_count, 1)

if __name__ == '__main__':
    unittest.main()