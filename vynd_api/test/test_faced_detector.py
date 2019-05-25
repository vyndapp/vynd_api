from typing import List, Tuple
import unittest
import numpy as np

from .test_utils import get_img_from_filename, url_to_img
from ..facedetection.bounding_box import BoundingBox
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.image_face_detector import ImageFaceDetector
from ..entities.keyframe import KeyFrame


class TestFacedDetector(unittest.TestCase):

    def setUp(self):
        self.non_rgb_input_url: str = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/1200px-Checkerboard_pattern.svg.png'
        self.valid_input_path: str = 'resources/faced.jpg'

        self.valid_input_img: np.array = get_img_from_filename(self.valid_input_path)
        self.non_rgb_input_img: np.ndarray = url_to_img(self.non_rgb_input_url)

        self.dummy_video_id = "dummy_video_id"
        self.dummy_keyframe_id = "dummy_keyframe_id"

        self.epsilon: float = 0.0001
        self.faced_detector: ImageFaceDetector = FacedDetector(minimum_confidence = 0.9, 
                                                               offset_value = 20,
                                                               pad_value=10)

    def test_non_rgb_input_status(self):
        keyframe = KeyFrame(self.non_rgb_input_img)
        keyframe.video_id = self.dummy_video_id
        keyframe.keyframe_id = self.dummy_keyframe_id
        detection_result: FaceDetectionResults = self.faced_detector.detect(keyframe)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.FAIL_NON_RGB_INPUT
        self.assertEqual(detection_result.status, expected_status)

    def test_non_rgb_input_bboxes(self):
        keyframe = KeyFrame(self.non_rgb_input_img)
        keyframe.video_id = self.dummy_video_id
        keyframe.keyframe_id = self.dummy_keyframe_id
        detection_result: FaceDetectionResults = self.faced_detector.detect(keyframe)
        expected_detected_faces = None
        self.assertEqual(detection_result.detected_faces, expected_detected_faces)

    def test_valid_input_status(self):
        keyframe = KeyFrame(self.valid_input_img)
        keyframe.video_id = self.dummy_video_id
        keyframe.keyframe_id = self.dummy_keyframe_id
        detection_result: FaceDetectionResults = self.faced_detector.detect(keyframe)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.SUCCESS
        self.assertEqual(detection_result.status, expected_status)
    
    def test_valid_input_bboxes(self):
        keyframe = KeyFrame(self.valid_input_img)
        keyframe.video_id = self.dummy_video_id
        keyframe.keyframe_id = self.dummy_keyframe_id
        detection_result: FaceDetectionResults = self.faced_detector.detect(keyframe)

        expected_bboxes: List[BoundingBox] = [BoundingBox((84, 154, 162, 238), 0.9987294),
                                              BoundingBox((186, 187, 268, 271), 0.9990884),
                                              BoundingBox((291, 202, 363, 282), 0.9980178),
                                              BoundingBox((374, 186, 446, 272), 0.99325585),
                                              BoundingBox((485, 183, 561, 265), 0.9943262)]

        self.assertEqual(len(expected_bboxes), len(detection_result.detected_faces))
                                              
        for i in range(len(expected_bboxes)):
            self.assertEqual(detection_result.detected_faces[i].bbox.coordinates, expected_bboxes[i].coordinates)

        for i in range(len(expected_bboxes)):
            actual_conf = detection_result.detected_faces[i].bbox.confidence
            expected_conf = expected_bboxes[i].confidence
            self.assertAlmostEqual(actual_conf, expected_conf, delta = self.epsilon)


    def test_valid_input_cropped_faces(self):
        keyframe = KeyFrame(self.valid_input_img)
        keyframe.video_id = self.dummy_video_id
        keyframe.keyframe_id = self.dummy_keyframe_id
        detection_result: FaceDetectionResults = self.faced_detector.detect(keyframe)

        expected_bboxes: List[BoundingBox] = [BoundingBox((84, 154, 162, 238), 0.9987294),
                                              BoundingBox((186, 187, 268, 271), 0.9990884),
                                              BoundingBox((291, 202, 363, 282), 0.9980178),
                                              BoundingBox((374, 186, 446, 272), 0.99325585),
                                              BoundingBox((485, 183, 561, 265), 0.9943262)]
        
        def get_shape(bbox: BoundingBox) -> Tuple[(int, int, int)]:
            (x_upperleft, y_upperleft, x_lowerright, y_lowerright) = bbox.coordinates
            return (y_lowerright - y_upperleft + 1, x_lowerright - x_upperleft + 1, 3)
        expected_shapes: List[Tuple[(int, int, int)]] = list(map(get_shape, expected_bboxes))

        self.assertEqual(len(expected_shapes), len(detection_result.detected_faces))

        for i, _ in enumerate(expected_shapes):
            self.assertEqual(detection_result.detected_faces[i].image.shape, expected_shapes[i])
       
if __name__ == '__main__':
    unittest.main()
    