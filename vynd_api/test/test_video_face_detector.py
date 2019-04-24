
from typing import List, Tuple

import requests
import unittest

from ..facedetection.bounding_box import BoundingBox
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..entities.image import KeyFrame
from .test_utils import url_to_base64, get_img_from_filename, filename_to_base64

class TestVideoFaceDetector(unittest.TestCase):

    def setUp(self):
        faced: ImageFaceDetector = FacedDetector()
        self.video_faced: VideoFaceDetector = VideoFaceDetector(faced)
        hosted_images: List[str] = ['https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/1200px-Checkerboard_pattern.svg.png',
                                    'https://collectionimages.npg.org.uk/std/mw198888/James-Martineau.jpg']
        local_images: List[str] = ['faced.jpg']

        self.epsilon: float = 0.0001
        self.key_frames = []

        for img_url in hosted_images:
            key_frame = KeyFrame(url_to_base64(img_url))
            self.key_frames.append(key_frame)
        
        for img_path in local_images:
            key_frame = KeyFrame(get_img_from_filename(img_path))
            self.key_frames.append(key_frame)

    def test_video_face_detector(self):
        expected_bboxes: List[BoundingBox] = [BoundingBox((76, 144, 154, 230), 0.99901056),
                                              BoundingBox((177, 181, 257, 263), 0.99876785),
                                              BoundingBox((281, 193, 353, 271), 0.99876535),
                                              BoundingBox((357, 175, 427, 259), 0.9876309),
                                              BoundingBox((474, 175, 552, 255), 0.9892281)]
        expected_results = [FaceDetectionResults(FaceDetectionStatus.FAIL_NON_RGB_INPUT, None),
                            FaceDetectionResults(FaceDetectionStatus.FAIL_NON_EQUAL_DIMS, None),
                            FaceDetectionResults(FaceDetectionStatus.SUCCESS, expected_bboxes)] 
    
        detection_results: List[FaceDetectionResults] = self.video_faced.get_detected_results(key_frames = self.key_frames)
        
        self.assertEqual(len(detection_results), len(expected_results))                 

        for i in range(len(detection_results)):
            self.assertEqual(detection_results[i].status, expected_results[i].status)

            if detection_results[i].detected_faces != None:
                self.assertEqual(len(detection_results[i].detected_faces), len(expected_results[i].detected_faces))
                
                for j in range(len(detection_results[i].detected_faces)):
                    self.assertEqual(detection_results[i].detected_faces[j].bbox.coordinates, expected_bboxes[j].coordinates)
                    self.assertAlmostEqual(detection_results[i].detected_faces[j].bbox.confidence, expected_bboxes[j].confidence, delta = self.epsilon)
            else:
                self.assertEqual(detection_results[i].detected_faces, None)

if __name__ == '__main__':
    unittest.main()