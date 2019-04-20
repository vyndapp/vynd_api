from typing import List
import unittest
from PIL import Image

import numpy as np
import requests

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
        self.valid_input_img: np.array = get_img_from_filename('faced.png')
        self.epsilon: float = 0.0001
        self.faced_detector: ImageFaceDetector = FacedDetector()

    def test_non_rgb_input_status(self):
        np_img = url_to_img(self.non_rgb_input_url)
        detection_result: FaceDetectionResults = self.faced_detector.detect(np_img)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.FAIL_NON_RGB_INPUT
        self.assertEqual(detection_result.status, expected_status)

    def test_non_rgb_input_bboxes(self):
        np_img = url_to_img(self.non_rgb_input_url)
        detection_result: FaceDetectionResults = self.faced_detector.detect(np_img)
        expected_bboxes = None
        self.assertEqual(detection_result.bboxes, expected_bboxes)

    def test_non_equal_dims_status(self):
        np_img = url_to_img(self.non_equal_dims_url)
        detection_result: FaceDetectionResults = self.faced_detector.detect(np_img)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.FAIL_NON_EQUAL_DIMS
        self.assertEqual(detection_result.status, expected_status)

    def test_non_equal_dims_bboxes(self):
        np_img = url_to_img(self.non_equal_dims_url)
        detection_result: FaceDetectionResults = self.faced_detector.detect(np_img)
        expected_bboxes = None
        self.assertEqual(detection_result.bboxes, expected_bboxes)

    def test_valid_input_status(self):
        np_img = get_img_from_filename('faced.png')
        detection_result: FaceDetectionResults = self.faced_detector.detect(np_img)
        expected_status: FaceDetectionStatus = FaceDetectionStatus.SUCCESS
        self.assertEqual(detection_result.status, expected_status)
    
    def test_valid_input_bboxes(self):
        np_img = get_img_from_filename('faced.png')
        detection_result: FaceDetectionResults = self.faced_detector.detect(np_img)

        expected_bboxes: List[BoundingBox] = [BoundingBox((76, 144, 154, 230), 0.99901056),
                                              BoundingBox((177, 181, 257, 263), 0.99876785),
                                              BoundingBox((281, 193, 353, 271), 0.99876535),
                                              BoundingBox((357, 175, 427, 259), 0.9876309),
                                              BoundingBox((474, 175, 552, 255), 0.9892281)]
                                              
        for i in range(len(expected_bboxes)):
            self.assertEqual(detection_result.bboxes[i].coordinates, expected_bboxes[i].coordinates)
        for i in range(len(expected_bboxes)):
            actual_conf = detection_result.bboxes[i].confidence
            expected_conf = expected_bboxes[i].confidence
            self.assertAlmostEqual(actual_conf, expected_conf, delta = self.epsilon)


if __name__ == '__main__':
    unittest.main()