
import dlib
import os
import cv2
import numpy as np

from PIL import Image
from typing import List
from dlib import rectangle
from pathlib import Path
from imutils.face_utils import FaceAligner

class FaceAlignment:
    
    def __init__(self):
        cur_dir = os.path.dirname(__file__)
        shape_predictor_path = Path(cur_dir, '../../models/landmark_detector/shape_predictor_5_face_landmarks.dat')
        self.__landmarks_detector = dlib.shape_predictor(str(shape_predictor_path))
        self.__desired_face_width = 200
        self.__cropping_ratio = 0.79
        self.__face_aligner = FaceAligner(self.__landmarks_detector, 
                                          desiredFaceWidth=self.__desired_face_width)
        
    def get_aligned_faces(self, original_image: np.ndarray, rectangles: List[rectangle]):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        aligned_faces = []
        
        for rect in rectangles:
            aligned_face = self.__face_aligner.align(original_image, gray_image, rect)
            
            new_start_y, new_end_y = 0, int(self.__desired_face_width * self.__cropping_ratio)
            new_start_x = int((1. - self.__cropping_ratio) * self.__desired_face_width // 2)
            new_end_x = int(self.__desired_face_width - new_start_x)
            
            aligned_face = aligned_face[new_start_y:new_end_y, new_start_x:new_end_x, :]
            aligned_faces.append(aligned_face)
            
        return aligned_faces