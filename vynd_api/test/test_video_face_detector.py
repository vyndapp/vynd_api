
from typing import List, Tuple
from PIL import Image

import requests
import unittest

from ..facedetection.bounding_box import BoundingBox
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..entities.keyframe import KeyFrame
from .test_utils import url_to_base64, get_img_from_filename, save_img

class TestVideoFaceDetector(unittest.TestCase):

    def setUp(self):
        faced: ImageFaceDetector = FacedDetector(minimum_confidence=0.9, 
                                                 offset_value=20,
                                                 pad_value=10)
        self.video_faced: VideoFaceDetector = VideoFaceDetector(faced)
        hosted_images: List[str] = ['https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/1200px-Checkerboard_pattern.svg.png']
        local_images: List[str] = ['resources/faced.jpg']

        self.epsilon: float = 0.0001
        self.key_frames = []

        self.dummy_video_id = "dummy_video_id"
        self.dummy_keyframe_id = "dummy_keyframe_id"

        for img_url in hosted_images:
            key_frame = KeyFrame(url_to_base64(img_url), self.dummy_video_id)
            key_frame.keyframe_id = self.dummy_keyframe_id
            self.key_frames.append(key_frame)
        
        for img_path in local_images:
            key_frame = KeyFrame(get_img_from_filename(img_path), self.dummy_video_id)
            key_frame.keyframe_id = self.dummy_keyframe_id
            self.key_frames.append(key_frame)

    def test_video_face_detector(self):
        expected_bboxes: List[BoundingBox] = [BoundingBox((84, 154, 162, 238), 0.9987294),
                                              BoundingBox((186, 187, 268, 271), 0.9990884),
                                              BoundingBox((291, 202, 363, 282), 0.9980178),
                                              BoundingBox((374, 186, 446, 272), 0.99325585),
                                              BoundingBox((485, 183, 561, 265), 0.9943262)]
        expected_results = [FaceDetectionResults(self.dummy_video_id, 
                                                 self.dummy_keyframe_id, 
                                                 FaceDetectionStatus.FAIL_NON_RGB_INPUT, 
                                                 None),
                            FaceDetectionResults(self.dummy_video_id,  
                                                 self.dummy_keyframe_id, 
                                                 FaceDetectionStatus.SUCCESS, 
                                                 expected_bboxes)] 
    
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