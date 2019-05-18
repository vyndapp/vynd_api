
from ..data import CLIENT
from ..data.video_collection import VideoCollection
from ..entities.video import Video

import unittest

class TestVideoCollection(unittest.TestCase):

    def setUp(self):
        self.video_collection = VideoCollection(collection=CLIENT.vynd_db_test.video_collection)

    def test_video_collection(self):
        video_id = self.video_collection.insert_new_video(Video(key_frames = []))
        
        keyframes = ["kf1", "kf2", "kf3", "kf4"]

        for kf_id in keyframes:
            result = self.video_collection.add_keyframe(video_id, kf_id)
            self.assertTrue(result)
        
        faces = ["f1", "f2", "f3", "f4", "f1", "f2"]

        for f_id in faces:
            result = self.video_collection.add_face(video_id, f_id)
            self.assertTrue(result)

        expected_number_of_keyframes = 4
        expected_number_of_faces = 4

        video = self.video_collection.get_video_by_id(video_id)
        self.assertEqual(len(video["keyframes_ids"]), expected_number_of_keyframes)
        self.assertEqual(len(video["faces_ids"]), expected_number_of_faces)

if __name__ == '__main__':
    unittest.main()