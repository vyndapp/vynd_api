
from typing import List
from flask_api import status

import numpy as np
import unittest
import requests
import json
import base64

from ..entities.keyframe import KeyFrame
from ..videoprocessing.video_processor import VideoProcessor
from .test_utils import get_all_local_images
from ..utils.image_utils import rgb_to_base64
from ..data.video_collection import VideoCollection
from ..data.face_collection import FaceCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..data import CLIENT
from ..data.db_utils import np_to_binary, binary_to_b64

class TestProcessKeyFrames(unittest.TestCase):

    def setUp(self):
        self.current_host_IP = 'http://127.0.0.5:5000/api'

        self.video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)
        self.face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)
        self.keyframe_collection = KeyFrameCollection(CLIENT.vynd_db_test.keyframe_collection)

        self.key_frames: KeyFrame = []
        local_images: List[np.ndarray] = get_all_local_images('resources/')
    
        for img in local_images:
            key_frame = KeyFrame(img)
            self.key_frames.append(key_frame)

    def test_invalid_video_id(self):
        base64_images = [keyframe.base64_image.decode() for keyframe in self.key_frames]
        json_data = {"video_id": "invalid_video_id",
                     "base64_images": json.dumps(base64_images)}
        result = self.__make_post_request(json_data)
        self.assertEqual(result, status.HTTP_400_BAD_REQUEST)   

    def test_invalid_keyframes_type(self):
        video_id = self.video_collection.insert_new_video()
        base64_images = [12 for keyframe in self.key_frames]
        json_data = {"video_id": video_id,
                     "base64_images": json.dumps(base64_images)}
        result = self.__make_post_request(json_data)
        self.assertEqual(result, status.HTTP_406_NOT_ACCEPTABLE)
        self.video_collection.delete_video(video_id)

    def test_missing_form(self):
        video_id = self.video_collection.insert_new_video()
        base64_images = [keyframe.base64_image.decode() for keyframe in self.key_frames]
        json_data = {"base64_images": json.dumps(base64_images)}
        result = self.__make_post_request(json_data)
        self.assertEqual(result, status.HTTP_400_BAD_REQUEST)
        self.video_collection.delete_video(video_id)

    def test_success(self):
        video_id = self.video_collection.insert_new_video()
        base64_images = [keyframe.base64_image.decode() for keyframe in self.key_frames]
        json_data = {"video_id": video_id,
                     "base64_images": json.dumps(base64_images)}
                     
        result = self.__make_post_request(json_data)

        self.assertEqual(result, status.HTTP_200_OK)
        self.assertEqual(self.face_collection.get_number_of_records(), 6)
        self.assertEqual(self.keyframe_collection.get_number_of_records(), 4)
        self.assertEqual(self.video_collection.get_number_of_records(), 1)

        self.face_collection.delete_all_faces()
        self.keyframe_collection.delete_all_keyframes()
        self.video_collection.delete_all_videos()


    def __make_post_request(self, json_data):
        try:
            headers = {'Content-type': 'application/json', 'Connection':'close'}
            response = requests.post(url=self.current_host_IP + '/process-keyframes', 
                                     json=json_data,
                                     headers=headers)
            return response.status_code                                   
        except:
            return None

if __name__ == '__main__':
    unittest.main()