
import numpy as np

from typing import List

from ..videoprocessing.video_processor import VideoProcessor
from ..videoprocessing.video_processing_results import VideoProcessingResult

from ..data.face_collection import FaceCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..data.video_collection import VideoCollection
from ..data import CLIENT

from ..entities.keyframe import KeyFrame
from ..entities.video import Video
from ..entities.user import User

from .test_utils import get_all_local_images, save_img

import unittest

class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        self.video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)
        self.face_collection = VideoCollection(CLIENT.vynd_db_test.face_collection)
        self.keyframe_collection = VideoCollection(CLIENT.vynd_db_test.keyframe_collection)

        self.key_frames: List[KeyFrame] = []
        local_images: List[np.ndarray] = get_all_local_images('resources/')

        for img in local_images:
            key_frame = KeyFrame(img)
            self.key_frames.append(key_frame)

# TODO: implement a test script for processor
    def test_video_processor(self):
        video_id = self.video_collection.insert_new_video()

        video_processor = VideoProcessor()
        processing_result: VideoProcessingResult = video_processor.process(video_id, self.key_frames)

        self.assertEqual(processing_result, VideoProcessingResult.SUCCESS)
        self.assertEqual(self.video_collection.get_number_of_records(), 1)
        self.assertEqual(self.face_collection.get_number_of_records(), 6)
        self.assertEqual(self.keyframe_collection.get_number_of_records(), 4)

        self.video_collection.delete_all_videos()
        self.face_collection.delete_all_videos()
        self.keyframe_collection.delete_all_videos()

if __name__ == '__main__':
    unittest.main()