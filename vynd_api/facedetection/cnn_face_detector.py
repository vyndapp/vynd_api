
import dlib
import cv2
import os

import numpy as np

from typing import List
from dlib import rectangle
from imutils.face_utils import rect_to_bb
from pathlib import Path

from ..entities.keyframe import KeyFrame
from .image_face_detector import ImageFaceDetector
from .facealignment.face_alignment import FaceAlignment
from .face_detection_results import FaceDetectionResults
from .face_detection_status import FaceDetectionStatus
from .bounding_box import BoundingBox
from .face_detection_results import DetectedFace

class CNNDlibDetector(ImageFaceDetector):
    
    def __init__(self, offset_value=10):
        cnn_weights_path = '../models/human_face_detector/mmod_human_face_detector.dat'
        cur_dir = os.path.dirname(__file__)
        cnn_model_path = str(Path(cur_dir, cnn_weights_path))
        self.__face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
        self.__offset_value = offset_value
        self.__face_aligner = FaceAlignment()
        self.__expected_n_channels = 3
        
    def detect(self, keyframe: KeyFrame) -> FaceDetectionResults:
        """
            Takes as input a KeyFrame entity and returns the FaceDetectionResults (status, bounding boxes) for that specific image:
            - image: must be a 3d numpy array (RGB image)
        """
        
        channels: np.int32 = keyframe.image.shape[2]

        if(channels != self.__expected_n_channels):
            print('nooo')
            return FaceDetectionResults(keyframe_id=keyframe.keyframe_id,
                                        video_id=keyframe.video_id,
                                        status=FaceDetectionStatus.FAIL_NON_RGB_INPUT)
        else:
            image = np.array(keyframe.image)
            image.setflags(write=True)
            print('processing...')
            faces_cnn = self.__face_detector(image, 2)
            print('done!')
            rectangles = [face.rect for face in faces_cnn]
            cropped_images = []
            for rect in rectangles:
                print('found rect!')
                (x, y, w, h) = rect_to_bb(rect)
                x_upperleft: int = max(0, x)
                y_upperleft: int = max(0, y) 
                x_lowerright: int = x + w 
                y_lowerright: int = y + h 
                cropped_images.append(image[y_upperleft: y_lowerright, x_upperleft:x_lowerright, :])
            
            aligned_faces = self.__face_aligner.get_aligned_faces(image, rectangles)
            
            detected_faces: List[DetectedFace] = list(map(lambda image, aligned_image: DetectedFace(image=image, aligned_image=aligned_image), cropped_images, aligned_faces))
            
            return FaceDetectionResults(detected_faces=detected_faces,
                                        keyframe_id=keyframe.keyframe_id,
                                        video_id=keyframe.video_id,
                                        status=FaceDetectionStatus.SUCCESS)
            
            
            
            