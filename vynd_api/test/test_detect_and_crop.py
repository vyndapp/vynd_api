
from typing import List

import numpy as np
import unittest

from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..entities.keyframe import KeyFrame
from .test_utils import save_img, get_all_local_images

class TestDetectAndCrop(unittest.TestCase):

    def setUp(self):
        # detector
        faced: ImageFaceDetector = FacedDetector(minimum_confidence = 0.8, offset_value = 0)
        self.video_faced: VideoFaceDetector = VideoFaceDetector(faced)
        # keyframes
        self.key_frames: KeyFrame = []
        local_images: List[np.ndarray] = get_all_local_images('resources/')
        
        for img in local_images:
            key_frame = KeyFrame(img)
            self.key_frames.append(key_frame)


    def test_detect_and_crop_from_local(self):
        detection_results: List[FaceDetectionResults] = self.video_faced.get_detected_results(key_frames = self.key_frames)

        for i in range(len(detection_results)):
            if detection_results[i].detected_faces != None:
                for j in range(len(detection_results[i].detected_faces)):
                    save_img('detected_faces/', detection_results[i].detected_faces[j].image, str(i) + str(j) + '.png')

if __name__ == '__main__':
    unittest.main()