
import numpy as np

from typing import List

from ..videoprocessing.video_processor import VideoProcessor
from ..videoprocessing.video_processing_results import VideoProcessingResult

from ..data.face_collection import FaceCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..data.video_collection import VideoCollection

from ..entities.keyframe import KeyFrame
from ..entities.video import Video
from ..entities.user import User

from .test_utils import get_all_local_images, save_img

import unittest

class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        dummy_video_id = "dummy_video_id"
        dummy_keyframe_id = "dummy_keyframe_id"

        # get keyframes
        key_frames: List[KeyFrame] = []
        local_images: List[np.ndarray] = get_all_local_images('resources/')
        i = 1
        for img in local_images:
            key_frame = KeyFrame(img)
            i += 1
            key_frames.append(key_frame)

        self.video = Video(key_frames=key_frames)

# TODO: implement a test script for processor
    def test_video_processor(self):
        video_processor = VideoProcessor(self.video, User(id_="dummy_user_id"))
        processing_result: VideoProcessingResult = video_processor.process()
        self.assertEqual(processing_result, VideoProcessingResult.SUCCESS)

if __name__ == '__main__':
    unittest.main()