
import dlib
import cv2

import numpy as np

from typing import List
from dlib import rectangle
from imutils.face_utils import rect_to_bb

from ..entities.keyframe import KeyFrame
from .image_face_detector import ImageFaceDetector
from .facealignment.face_alignment import FaceAlignment
from .face_detection_results import FaceDetectionResults
from .face_detection_status import FaceDetectionStatus
from .bounding_box import BoundingBox
from .face_detection_results import DetectedFace

class HogDetector(ImageFaceDetector):
    
    def __init__(self, offset_value=10):
        self.__offset_value = offset_value
        self.__face_detector = dlib.get_frontal_face_detector()
        self.__face_aligner = FaceAlignment()
        self.__expected_n_channels = 3
        
    def detect(self, keyframe: KeyFrame) -> FaceDetectionResults:
        """
            Takes as input a KeyFrame entity and returns the FaceDetectionResults (status, bounding boxes) for that specific image:
            - image: must be a 3d numpy array (RGB image)
        """
        
        channels: np.int32 = keyframe.image.shape[2]
        
        if(channels != self.__expected_n_channels):
            return FaceDetectionResults(keyframe_id=keyframe.keyframe_id,
                                        video_id=keyframe.video_id,
                                        status=FaceDetectionStatus.FAIL_NON_RGB_INPUT)
        else:
            gray_image = cv2.cvtColor(keyframe.image, cv2.COLOR_BGR2GRAY)
            rectangles = self.__face_detector(gray_image, 3)
            
            cropped_images = []
            for rect in rectangles:
                (x, y, w, h) = rect_to_bb(rect)
                x_upperleft: int = max(0, x)
                y_upperleft: int = max(0, y) 
                x_lowerright: int = x + w 
                y_lowerright: int = y + h 
                cropped_images.append(keyframe.image[y_upperleft: y_lowerright, x_upperleft:x_lowerright, :])
            
            aligned_faces = self.__face_aligner.get_aligned_faces(keyframe.image, rectangles)
            
            detected_faces: List[DetectedFace] = list(map(lambda image, aligned_image: DetectedFace(image=image, aligned_image=aligned_image), cropped_images, aligned_faces))
            
            return FaceDetectionResults(detected_faces=detected_faces,
                                        keyframe_id=keyframe.keyframe_id,
                                        video_id=keyframe.video_id,
                                        status=FaceDetectionStatus.SUCCESS)
            
            
            
            