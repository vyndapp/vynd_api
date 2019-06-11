
import dlib
import cv2
import os

import numpy as np

from pathlib import Path
from typing import List
from dlib import rectangle

from ..entities.keyframe import KeyFrame
from .image_face_detector import ImageFaceDetector
from .facealignment.face_alignment import FaceAlignment
from .face_detection_results import FaceDetectionResults
from .face_detection_status import FaceDetectionStatus
from .bounding_box import BoundingBox
from .face_detection_results import DetectedFace
from ..utils import vggface2_utlis, image_utils
from ..test.test_utils import save_img

class YOLOv3Detector(ImageFaceDetector):
    
    # TODO: check if model is downloaded or not
    def __init__(self, offset_value=20):
        yolov3_weights_path = self.__get_real_path('../models/yolov3/yolov3-wider_16000.weights')
        yolov3_cfg_path = self.__get_real_path('../models/yolov3/yolov3-face.cfg')
        self.__yolov3 = self.__init_detector(weights_path=yolov3_weights_path, 
                                             cfg_path=yolov3_cfg_path)
        self.__offset_value = offset_value
        self.__face_aligner = FaceAlignment()
        self.__default_dims = (416, 416)
        self.__expected_n_channels = 3
        self.__CONF_THRESHOLD = 0.6
        self.__NMS_THRESHOLD = 0.4
                
    def __init_detector(self, weights_path: str, cfg_path: str):
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    def __get_real_path(self, path):
        cur_dir = os.path.dirname(__file__)
        real_path = str(Path(cur_dir, path))
        return real_path
    
    def __pad_image(self, image: np.ndarray, pad_value: int) -> np.ndarray:
        """
        Padds the input image with the specified value with a constant value=255 in the width and height only.
        """
        return np.pad(image, 
                      ((pad_value, pad_value), (pad_value, pad_value), (0, 0)), 
                      mode='constant', 
                      constant_values=0)
        
    def detect(self, keyframe: KeyFrame) -> FaceDetectionResults:
        """
            Takes as input a KeyFrame entity and returns the FaceDetectionResults (status, bounding boxes) for that specific image:
            - image: must be a 3d numpy array (RGB image)
        """
        padded_image = self.__pad_image(keyframe.image, 20)
        
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(keyframe.image,                                     
                                     1 / 255, 
                                     self.__default_dims,
                                     [0, 0, 0], 
                                     1, 
                                     crop=False)

        # Sets the input to the network
        self.__yolov3.setInput(blob)
        
        outs = self.__yolov3.forward(self.__get_outputs_names())

        faces = self.__post_process(keyframe.image, outs, self.__CONF_THRESHOLD, self.__NMS_THRESHOLD)

        cropped_images = []
        rectangles = []
        for face in faces:
            x, y, w, h = face[0], face[1], face[2], face[3]
            x_upperleft = max(0, x - self.__offset_value)
            y_upperleft = max(0, y - self.__offset_value)
            x_lowerright = min(keyframe.image.shape[1], x + w + self.__offset_value)
            y_lowerright = min(keyframe.image.shape[0], y + h + self.__offset_value)
            cropped_images.append(keyframe.image[y_upperleft:y_lowerright, x_upperleft:x_lowerright, :])
            rectangles.append(rectangle(x, y, x + w, y + h))
            
        aligned_faces = self.__face_aligner.get_aligned_faces(keyframe.image, rectangles)
        
        detected_faces: List[DetectedFace] = list(map(lambda image, aligned_image: DetectedFace(image=image, aligned_image=aligned_image), cropped_images, aligned_faces))
        
        return FaceDetectionResults(detected_faces=detected_faces,
                                    keyframe_id=keyframe.keyframe_id,
                                    video_id=keyframe.video_id,
                                    status=FaceDetectionStatus.SUCCESS)
    
    def __get_outputs_names(self):
        # Get the names of all the layers in the network
        layers_names = self.__yolov3.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected
        # outputs
        return [layers_names[i[0] - 1] for i in self.__yolov3.getUnconnectedOutLayers()]
    
    def __post_process(self, frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            final_boxes.append(box)
            left, top, _, _ = self.__refined_box(left, top, width, height)
        return final_boxes
    
    def __refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

        right = right + margin

        return left, top, right, bottom

yolov3 = YOLOv3Detector()
