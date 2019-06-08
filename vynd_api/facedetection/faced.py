
from typing import List, Tuple
from faced import FaceDetector
from dlib import rectangle

import numpy as np

from .image_face_detector import ImageFaceDetector
from .face_detection_results import FaceDetectionResults, DetectedFace
from .face_detection_status import FaceDetectionStatus
from .bounding_box import BoundingBox
from .facealignment.face_alignment import FaceAlignment
from ..entities.keyframe import KeyFrame

class FacedDetector(ImageFaceDetector):
    
    def __init__(self, minimum_confidence: float = 0.9, offset_value: float = 25, pad_value: int = 15):
        """
            Faced Face Detection Algorithm:
            - A single shot detection algorithm with CPU bound performance based on CNNs
            - minimum_confidence: the minimum score for any detected face to be counted as a real face
            - offset_value: offset added to the predicted bounding box to capture larger boxes that contains the whole face
            - pad_value: padding value added to the original keyframe image
        """
        self.__faced = FaceDetector()
        self.__expected_n_channels = 3
        self.__minimum_confidence = minimum_confidence
        self.__offset_value = offset_value
        self.__pad_value = pad_value
        self.__face_aligner = FaceAlignment()
    
    def detect(self, keyframe: KeyFrame) -> FaceDetectionResults:
        """
            Takes as input a KeyFrame entity and returns the FaceDetectionResults (status, bounding boxes) for that specific image:
            - image: must be a 3d numpy array (RGB image)
        """
        # square_image = self.__rectangle_to_square_image(keyframe.image)
        padded_image = self.__pad_image(keyframe.image, self.__pad_value)

        channels: np.int32 = keyframe.image.shape[2]
        
        if(channels != self.__expected_n_channels):
            return FaceDetectionResults(keyframe_id=keyframe.keyframe_id,
                                        video_id=keyframe.video_id,
                                        status=FaceDetectionStatus.FAIL_NON_RGB_INPUT)
        else:
            predicted_bboxes = self.__faced.predict(frame=padded_image, 
                                                    thresh=self.__minimum_confidence)

            final_bboxes: List[BoundingBox] = self.__preprocess_predicted_bboxes(predicted_bboxes=predicted_bboxes, 
                                                                                 image_width=keyframe.image.shape[1], 
                                                                                 image_height=keyframe.image.shape[0])

            face_images: List[np.array] = self.__get_face_images(image=padded_image, 
                                                                 final_bboxes=final_bboxes)
            
            aligned_face_images: List[np.array] = self.__align_faces(image=padded_image, 
                                                                     bboxes=final_bboxes)

            detected_faces: List[DetectedFace] = list(map(lambda image, aligned_image: DetectedFace(image=image, aligned_image=aligned_image), face_images, aligned_face_images))
            
            return FaceDetectionResults(detected_faces=detected_faces,
                                        keyframe_id=keyframe.keyframe_id,
                                        video_id=keyframe.video_id,
                                        status=FaceDetectionStatus.SUCCESS)

    def __pad_image(self, image: np.ndarray, pad_value: int) -> np.ndarray:
        """
        Padds the input image with the specified value with a constant value=255 in the width and height only.
        """
        return np.pad(image, 
                      ((pad_value, pad_value), (pad_value, pad_value), (0, 0)), 
                      mode='constant', 
                      constant_values=0)

    def __rectangle_to_square_image(self, image: np.ndarray) -> np.ndarray:
        """
        Transforms the input image to a sqaure-shaped image
        """
        image_width, image_height = image.shape[1], image.shape[0]
        if(image_height == image_width):
            return image
        elif(image_height > image_width):
            remaining = (image_height - image_width) // 2
            square_image = np.pad(image,
                                  ((0, 0), (remaining, remaining), (0, 0)),
                                  mode='constant',
                                  constant_values=0)
            return square_image
        else:
            remaining = (image_width - image_height) // 2
            square_image = np.pad(image,
                                  ((remaining, remaining), (0, 0), (0, 0)),
                                  mode='constant',
                                  constant_values=0)
            return square_image

        
    def __preprocess_predicted_bboxes(self, predicted_bboxes, image_width: int, image_height: int) -> List[BoundingBox]:
        final_bboxes: List[BoundingBox] = []
        
        for bbox in predicted_bboxes:
            x_center: int = bbox[0]
            y_center: int = bbox[1]
            width: int = bbox[2] // 2
            height: int = bbox[3] // 2
            confidence: float = bbox[4]
            
            x_upperleft: int = max(0, x_center - (width + self.__offset_value))
            y_upperleft: int = max(0, y_center - (height + self.__offset_value))
            x_lowerright: int = min(image_width - 1, x_center + (width + self.__offset_value))
            y_lowerright: int = min(image_height - 1, y_center + (height + self.__offset_value))
            
            coordinates: Tuple[int, int, int, int] = (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
            
            new_bbox: BoundingBox = BoundingBox(coordinates = coordinates, confidence = confidence)
            final_bboxes.append(new_bbox)
        
        return final_bboxes

    def __get_face_images(self, image: np.array, final_bboxes: List[BoundingBox]) -> List[np.array]:
        def get_face(bbox: BoundingBox) -> np.array:
            (x_upperleft, y_upperleft, x_lowerright, y_lowerright) = bbox.coordinates
            return image[y_upperleft: y_lowerright + 1, x_upperleft:x_lowerright + 1, :]
        return list(map(get_face, final_bboxes))
    
    def __align_faces(self, image: np.array, bboxes: List[BoundingBox]) -> List[np.array]:
        rectangles = self.__bbox_to_rect(image,bboxes=bboxes)
        return self.__face_aligner.get_aligned_faces(original_image=image,
                                                     rectangles=rectangles)
    
    def __bbox_to_rect(self, image,bboxes: List[BoundingBox]) -> List[rectangle]:
        rectangles = []
        
        for bbox in bboxes:
            x_upperleft, y_upperleft, x_lowerright, y_lowerright = bbox.coordinates
            rectangles.append(rectangle(x_upperleft, y_upperleft, x_lowerright, y_lowerright))
            
        return rectangles
        
