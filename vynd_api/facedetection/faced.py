from typing import List, Tuple
from faced import FaceDetector
import numpy as np

from .image_face_detector import ImageFaceDetector
from .face_detection_results import FaceDetectionResults, DetectedFace
from .face_detection_status import FaceDetectionStatus
from .bounding_box import BoundingBox

class FacedDetector(ImageFaceDetector):
    
    def __init__(self, minimum_confidence = 0.8, offset_value = 20):
        """
            Faced Face Detection Algorithm:
            - A single shot detection algorithm with CPU bound performance based on CNNs
            - minimum_confidence: the minimum score for any detected face to be counted as a real face
            - offset_value: offset added to the predicted bounding box to capture larger boxes that contains the whole face
        """
        self.__faced = FaceDetector()
        self.__expected_n_channels = 3
        self.__minimum_confidence = minimum_confidence
        self.__offset_value = offset_value
    
    def detect(self, image: np.ndarray) -> FaceDetectionResults:
        """
            Takes as input an image and returns the Face Detection Results (status, bounding boxes) for that specific image:
            - image: must be a numpy array of matching width and height, and three channels (RGB)
        """
        height: np.int32 = image.shape[0]
        width: np.int32 = image.shape[1]
        channels: np.int32 = image.shape[2]
        
        if(height != width):
            return FaceDetectionResults(status = FaceDetectionStatus.FAIL_NON_EQUAL_DIMS)
        elif(channels != self.__expected_n_channels):
            return FaceDetectionResults(status = FaceDetectionStatus.FAIL_NON_RGB_INPUT)
        else:
            predicted_bboxes = self.__faced.predict(frame = image, thresh = self.__minimum_confidence)
            final_bboxes: List[BoundingBox] = self.__preprocess_predicted_bboxes(predicted_bboxes)
            face_images: List[np.array] = self.__get_face_images(image, final_bboxes)

            detected_faces: List[DetectedFace] = list(map(lambda bbox, image: DetectedFace(bbox=bbox, image=image), final_bboxes, face_images))
            return FaceDetectionResults(detected_faces=detected_faces, status=FaceDetectionStatus.SUCCESS)
        
    def __preprocess_predicted_bboxes(self, predicted_bboxes) -> List[BoundingBox]:
        final_bboxes: List[BoundingBox] = []
        
        for bbox in predicted_bboxes:
            x_center: int = bbox[0]
            y_center: int = bbox[1]
            width: int = bbox[2] // 2
            height: int = bbox[3] // 2
            confidence: float = bbox[4]
            
            x_upperleft: int = x_center - (width + self.__offset_value)
            y_upperleft: int = y_center - (height + self.__offset_value)
            x_lowerright: int = x_center + (width + self.__offset_value)
            y_lowerright: int = y_center + (height + self.__offset_value)
            
            coordinates: Tuple[int, int, int, int] = (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
            
            new_bbox: BoundingBox = BoundingBox(coordinates = coordinates, confidence = confidence)
            final_bboxes.append(new_bbox)
        
        return final_bboxes

    def __get_face_images(self, image: np.array, final_bboxes: List[BoundingBox]) -> List[np.array]:
        def get_face(bbox: BoundingBox) -> np.array:
            (x_upperleft, y_upperleft, x_lowerright, y_lowerright) = bbox.coordinates
            return image[y_upperleft: y_lowerright + 1, x_upperleft:x_lowerright + 1, :]
        return list(map(get_face, final_bboxes))
