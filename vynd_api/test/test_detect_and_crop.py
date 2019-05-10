
from typing import List, Tuple
from PIL import Image
from pathlib import Path

import numpy as np
import os
import requests
import unittest

from ..facerecognition.video_face_recognizer import VideoFaceRecognizer
from ..facerecognition.image_face_recognizer import ImageFaceRecognizer
from ..facerecognition.vggface2 import VGGFaceRecognizer
from ..facerecognition.face_recognition_results import FaceRecognitionResults
from ..facedetection.bounding_box import BoundingBox
from ..facedetection.faced import FacedDetector
from ..facedetection.face_detection_results import FaceDetectionResults
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..entities.keyframe import KeyFrame
from ..entities.video import Video
from ..data.video_collection import VideoCollection
from ..data.keyframe_collection import KeyFrameCollection
from .test_utils import url_to_base64, get_img_from_filename, save_img, get_all_local_images

class TestDetectAndCrop(unittest.TestCase):

    def setUp(self):
        # detector
        faced: ImageFaceDetector = FacedDetector(minimum_confidence = 0.45, offset_value = 0)
        self.video_faced: VideoFaceDetector = VideoFaceDetector(faced)
        # recognizer
        vggface2: ImageFaceRecognizer = VGGFaceRecognizer()
        self.video_recognizer: VideoFaceRecognizer = VideoFaceRecognizer(vggface2)
        # data
        self.video_collection = VideoCollection()
        self.keyframe_collection = KeyFrameCollection()

        self.key_frames: KeyFrame = []
        local_images: List[np.ndarray] = get_all_local_images('resources/')
        print(len(local_images))
        
        for img in local_images:
            key_frame = KeyFrame(img)
            self.key_frames.append(key_frame)

        # # save entities in DB
        # video = Video(length_in_seconds=10, key_frames=self.key_frames)
        # self.video_id = self.video_collection.insert_video(video=video)
        # self.keyframe_ids = []

        # for i in range(len(video.key_frames)):
        #     keyframe_id = self.keyframe_collection.insert_keyframe(video_id=self.video_id, 
        #                                                            order_in_time=i, 
        #                                                            key_frame=video.key_frames[i])
        #     self.keyframe_ids.append(keyframe_id)

    def test_detect_and_crop_from_local(self):
        detection_results: List[FaceDetectionResults] = self.video_faced.get_detected_results(key_frames = self.key_frames)

        for i in range(len(detection_results)):
            if detection_results[i].detected_faces != None:
                for j in range(len(detection_results[i].detected_faces)):
                    save_img('detected_faces/', detection_results[i].detected_faces[j].image, str(i) + str(j) + '.png')
        
        recognition_results: FaceRecognitionResults = self.video_recognizer.recognize(detection_results)
        
        # for i in range(len(recognition_results)):
        #     for face in recognition_results[i].faces:
        #         print(face.features.shape)


if __name__ == '__main__':
    unittest.main()