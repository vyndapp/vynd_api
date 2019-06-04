import unittest

import numpy as np
from typing import List

from ..videoprocessing.video_processor import VideoProcessor
from ..videoprocessing.video_processing_results import VideoProcessingResult

from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder

from ..data.video_collection import VideoCollection
from ..data.face_collection import FaceCollection
from ..data.keyframe_collection import KeyFrameCollection

from ..data import CLIENT

from ..entities.keyframe import KeyFrame

from .test_utils import get_img_from_filename

class TestVideoProcessor(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestVideoProcessor, self).__init__(*args, **kwargs)
        self.video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)
        self.face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)
        self.keyframe_collection = KeyFrameCollection(CLIENT.vynd_db_test.keyframe_collection)
        self.video_processor = VideoProcessor(
            face_collection=CLIENT.vynd_db_test.face_collection,\
            keyframe_collection=CLIENT.vynd_db_test.key_frame_collection, \
            video_collection=CLIENT.vynd_db_test.video_collection)
        self.group_image_a = get_img_from_filename('resources/faced.jpg')
        self.group_image_b = get_img_from_filename('resources/1.jpg')
        self.video_collection.delete_all_videos()
        self.face_collection.delete_all_faces()
        self.keyframe_collection.delete_all_keyframes()
        
    def tearDown(self):
        self.video_collection.delete_all_videos()
        self.face_collection.delete_all_faces()
        self.keyframe_collection.delete_all_keyframes()

    def get_key_frames(self, images) -> List[KeyFrame]:
        return list(map(lambda img: KeyFrame(img), images))

    def test_new_faces_inserted(self):
        video_id = self.video_collection.insert_new_video()
        key_frames = self.get_key_frames([self.group_image_a])
        processing_result = self.video_processor.process(video_id, key_frames)
        self.assertEqual(processing_result, VideoProcessingResult.SUCCESS)
        self.assertEqual(self.face_collection.get_number_of_records(), 5)
        video = self.video_collection.get_video_by_id(video_id=video_id)
        faces_ids = video['faces_ids']
        self.assertEqual(len(set(faces_ids)), 5)

    def test_new_matches_found(self):
        # omar, bahi, gasser, yahya, hesham
        video_id_a = self.video_collection.insert_new_video()
        key_frames_a = self.get_key_frames([self.group_image_a])
        self.video_processor.process(video_id_a, key_frames_a)

        # omar, hazem, yahya, gasser in a group photo + hazem and omar inviduals
        image_zuma = get_img_from_filename('resources/zuma.jpg')
        image_omar = get_img_from_filename('resources/aligned_faces/o2.png')

        video_id_b = self.video_collection.insert_new_video()
        key_frames_b = self.get_key_frames([self.group_image_b, image_omar, image_zuma])
        self.video_processor.process(video_id_b, key_frames_b)

        self.assertEqual(self.face_collection.get_number_of_records(), 6)

        # assert faces are associated to video correctly
        video_b = self.video_collection.get_video_by_id(video_id=video_id_b)
        faces_ids_in_video_b = video_b['faces_ids']
        self.assertEqual(len(set(faces_ids_in_video_b)), 4)

        # assert videos are associated to faces correctly
        faces_videos_ids = self.face_collection.get_all_faces_video_ids()
        faces_videos_ids_lengths = list(map(lambda elem: len(elem['video_ids']), faces_videos_ids))
        self.assertCountEqual([1, 1, 1, 2, 2, 2], faces_videos_ids_lengths)

if __name__ == '__main__':
    unittest.main()