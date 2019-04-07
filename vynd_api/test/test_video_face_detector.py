
from typing import List

import base64
import requests
import unittest

from ..facedetection.bounding_box import BoundingBox
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..entities.image import KeyFrame

class TestVideoFaceDetector(unittest.TestCase):

    def setUp(self):
        faced: ImageFaceDetector = FacedDetector()
        self.video_faced: VideoFaceDetector = VideoFaceDetector(faced)
        images: List[str] = ['https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/1200px-Checkerboard_pattern.svg.png',
                             'https://collectionimages.npg.org.uk/std/mw198888/James-Martineau.jpg',
                             'https://doc-0o-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/l3mvvrb90ho9h0rkstgnrbr7thv7eo3d/1554645600000/13677457773113205383/*/1oe18qjXKUY_jQBFUixQUQ5wbKEkM_23B']
        
        self.epsilon: float = 0.0001
        self.key_frames = []
        for img_url in images:
            response_content = requests.get(img_url).content
            key_frame = KeyFrame(base64.b64encode(response_content))
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

            if detection_results[i].bboxes != None:
                self.assertEqual(len(detection_results[i].bboxes), len(expected_results[i].bboxes))
                
                for j in range(len(detection_results[i].bboxes)):
                    self.assertEqual(detection_results[i].bboxes[j].coordinates, expected_bboxes[j].coordinates)
                    self.assertAlmostEqual(detection_results[i].bboxes[j].confidence, expected_bboxes[j].confidence, delta = self.epsilon)
            else:
                self.assertEqual(detection_results[i].bboxes, None)

if __name__ == '__main__':
    unittest.main()