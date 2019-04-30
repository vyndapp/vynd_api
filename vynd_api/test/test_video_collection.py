from typing import List

from ..data.video_collection import VideoCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..entities.video import Video
from ..entities.image import KeyFrame
from .test_utils import get_all_local_images

import numpy as np
import unittest

class TestVideoCollection(unittest.TestCase):

    def setUp(self):
        local_images: List[np.ndarray] = get_all_local_images()
        self.key_frames = []
        
        for img in local_images:
            key_frame = KeyFrame(img)
            self.key_frames.append(key_frame)

        self.video_collection = VideoCollection()
        self.keyframe_collection = KeyFrameCollection()

    def test_video_collection(self):
        video = Video(length_in_seconds = 4, key_frames = self.key_frames)
        video_id = self.video_collection.insert_video(video)
        self.assertEqual(len(video_id), 24)

if __name__ == '__main__':
    unittest.main()