from typing import List, Tuple
import unittest
import numpy as np

from .test_utils import get_img_from_filename, url_to_img
from ..facedetection.bounding_box import BoundingBox
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.image_face_detector import ImageFaceDetector


class TestFacedDetector(unittest.TestCase):

    def setUp(self):
        self.non_rgb_input_url: str = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/1200px-Checkerboard_pattern.svg.png'
        self.non_equal_dims_url: str = 'https://collectionimages.npg.org.uk/std/mw198888/James-Martineau.jpg'
        self.valid_input_path: str = 'faced.jpg'

        self.valid_input_img: np.array = get_img_from_filename(self.valid_input_path)
        self.non_equal_dims_img: np.ndarray = url_to_img(self.non_equal_dims_url)
        self.non_rgb_input_img: np.ndarray = url_to_img(self.non_rgb_input_url)

        self.epsilon: float = 0.0001
        self.faced_detector: ImageFaceDetector = FacedDetector(minimum_confidence = 0.8, offset_value = 20)

    def test_non_rgb_input_status(self):
        detection_result: FaceDetectionResults = self.faced_detector.detect(self.non_rgb_input_img)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.FAIL_NON_RGB_INPUT
        self.assertEqual(detection_result.status, expected_status)

    def test_non_rgb_input_bboxes(self):
        detection_result: FaceDetectionResults = self.faced_detector.detect(self.non_rgb_input_img)
        expected_detected_faces = None
        self.assertEqual(detection_result.detected_faces, expected_detected_faces)

    # def test_non_equal_dims_status(self):
    #     detection_result: FaceDetectionResults = self.faced_detector.detect(self.non_equal_dims_img)
    #     expected_status: FaceDetectionStatus = FaceDetectionStatus.FAIL_NON_EQUAL_DIMS
    #     self.assertEqual(detection_result.status, expected_status)

    # def test_non_equal_dims_bboxes(self):
    #     detection_result: FaceDetectionResults = self.faced_detector.detect(self.non_equal_dims_img)
    #     expected_detected_faces = None
    #     self.assertEqual(detection_result.detected_faces, expected_detected_faces)

    def test_valid_input_status(self):
        detection_result: FaceDetectionResults = self.faced_detector.detect(self.valid_input_img)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.SUCCESS
        self.assertEqual(detection_result.status, expected_status)
    
    def test_valid_input_bboxes(self):
        detection_result: FaceDetectionResults = self.faced_detector.detect(self.valid_input_img)

        expected_bboxes: List[BoundingBox] = [BoundingBox((76, 144, 154, 230), 0.99901056),
                                              BoundingBox((177, 181, 257, 263), 0.99876785),
                                              BoundingBox((281, 193, 353, 271), 0.99876535),
                                              BoundingBox((357, 175, 427, 259), 0.9876309),
                                              BoundingBox((474, 175, 552, 255), 0.9892281)]

        self.assertEqual(len(expected_bboxes), len(detection_result.detected_faces))
                                              
        for i in range(len(expected_bboxes)):
            self.assertEqual(detection_result.detected_faces[i].bbox.coordinates, expected_bboxes[i].coordinates)

        for i in range(len(expected_bboxes)):
            actual_conf = detection_result.detected_faces[i].bbox.confidence
            expected_conf = expected_bboxes[i].confidence
            self.assertAlmostEqual(actual_conf, expected_conf, delta = self.epsilon)


    def test_valid_input_cropped_faces(self):
        detection_result: FaceDetectionResults = self.faced_detector.detect(self.valid_input_img)

        expected_bboxes: List[BoundingBox] = [BoundingBox((76, 144, 154, 230), 0.99901056),
                                              BoundingBox((177, 181, 257, 263), 0.99876785),
                                              BoundingBox((281, 193, 353, 271), 0.99876535),
                                              BoundingBox((357, 175, 427, 259), 0.9876309),
                                              BoundingBox((474, 175, 552, 255), 0.9892281)]
        
        def get_shape(bbox: BoundingBox) -> Tuple[(int, int, int)]:
            (x_upperleft, y_upperleft, x_lowerright, y_lowerright) = bbox.coordinates
            return (y_lowerright - y_upperleft + 1, x_lowerright - x_upperleft + 1, 3)
        expected_shapes: List[Tuple[(int, int, int)]] = list(map(get_shape, expected_bboxes))

        self.assertEqual(len(expected_shapes), len(detection_result.detected_faces))

        for i, _ in enumerate(expected_shapes):
            self.assertEqual(detection_result.detected_faces[i].image.shape, expected_shapes[i])
       
if __name__ == '__main__':
    unittest.main()
    