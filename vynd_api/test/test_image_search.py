import unittest
from typing import List

from ..videoprocessing.video_processor import VideoProcessor

from ..search.image_search import search_by_image
from ..data.video_collection import VideoCollection
from ..data.face_collection import FaceCollection

from ..data import CLIENT

from ..entities.keyframe import KeyFrame

from .test_utils import get_img_from_filename


class TestSearchByImage(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSearchByImage, self).__init__(*args, **kwargs)
        self.video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)
        self.face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)
        self.video_processor = VideoProcessor(
            face_collection=CLIENT.vynd_db_test.face_collection,\
            keyframe_collection=CLIENT.vynd_db_test.key_frame_collection, \
            video_collection=CLIENT.vynd_db_test.video_collection)
        self.group_image_a = get_img_from_filename('resources/faced.jpg')
        self.group_image_b = get_img_from_filename('resources/1.jpg')
        self.video_collection.delete_all_videos()
        self.face_collection.delete_all_faces()
        
    def tearDown(self):
        self.video_collection.delete_all_videos()
        self.face_collection.delete_all_faces()
        
    def get_key_frames(self, images) -> List[KeyFrame]:
        return list(map(lambda img: KeyFrame(img), images))

    def test_union_faces_found(self):
        # Insert Gasser, Yahya, Hesham, Omar, Bahi in video_a.
        video_id_a = self.video_collection.insert_new_video()
        key_frames = self.get_key_frames([self.group_image_a])
        self.video_processor.process(video_id_a, key_frames)

        # Search for an image containg Omar, Hazem, Yahya, Gasser.
        faces_ids = search_by_image(image=self.group_image_b, face_collection=CLIENT.vynd_db_test.face_collection) 
        self.assertEqual(len(faces_ids), 3)

    def test_no_faces_found(self):
        faces_ids = search_by_image(image=self.group_image_b, face_collection=CLIENT.vynd_db_test.face_collection)
        self.assertEqual(faces_ids, [])

if __name__ == '__main__':
    unittest.main()
    