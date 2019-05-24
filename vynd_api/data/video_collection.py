
from typing import Optional, List
from bson.objectid import ObjectId

from . import CLIENT
from ..entities.video import Video

class VideoCollection:
    def __init__(self, collection=CLIENT.vynd_db_test.video_collection):
        self.__collection = collection

    def insert_new_video(self, video: Video) -> str:
        return str(self.__collection.insert_one(
            {
                'is_processed': video.is_processed,
                'keyframes_ids': [],
                'faces_ids': []
            }
        ).inserted_id)

    def get_video_by_id(self, video_id: str):
        return self.__collection.find_one({'_id': ObjectId(video_id)})

    def add_keyframe(self, video_id: str, keyframe_id: str):
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$push': {'keyframes_ids': keyframe_id}})
        return (result.matched_count > 0)
    
    def add_face(self, video_id: str, face_id: str):
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$addToSet': {'faces_ids': face_id}})
        return (result.matched_count > 0)
    
    def update_status(self, video_id: str, new_status: bool):
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$set': {'is_processed': new_status}})
        return (result.matched_count > 0)

    def delete_all_videos(self):
        self.__collection.delete_many({})

    def get_all_video(self):
        self.__collection.find()

# vc = VideoCollection()
# vc.delete_all_videos()