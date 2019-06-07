from typing import List, Union
from bson.objectid import ObjectId
from pymongo.results import DeleteResult

from . import CLIENT

class VideoCollection:
    def __init__(self, collection=CLIENT.vynd_db.video_collection):
        self.__collection = collection

    def insert_new_video(self) -> str:
        """
        Inserts new video entity to DB and return the inserted video_id
        """
        return str(self.__collection.insert_one(
            {
                'is_processed': False,
                'keyframes_ids': [],
                'faces_ids': []
            }
        ).inserted_id)

    def get_video_by_id(self, video_id: str):
        """
        Params:
        - video_id: str
        Returns:
        - video_entity: dict
        """
        return self.__collection.find_one({'_id': ObjectId(video_id)})

    def get_faces(self, video_id: str):
        return list(self.__collection.find(filter={'_id': ObjectId(video_id)},
                    projection={'faces_ids': True, '_id': False}))

    def get_processed_videos(self):
        return list(self.__collection.find(filter={'is_processed': True},
                                           projection={'_id': True}))

    def add_keyframe(self, video_id: str, keyframe_id: str):
        """
        Params:
        - video_id: str
        - keyframe_id: str
        Returns:
        - insertion_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$push': {'keyframes_ids': keyframe_id}})
        return (result.matched_count > 0)
    
    def add_face(self, video_id: str, face_id: str):
        """
        Params:
        - video_id: str
        - face_id: str
        Returns:
        - insertion_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$addToSet': {'faces_ids': face_id}})
        return (result.matched_count > 0)
    
    def add_faces(self, video_id: str, face_ids: Union[str, List[str]]):
        """
        face_ids can be a single value or a list
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$push': {'faces_ids': {'$each': face_ids}}})
        return (result.matched_count > 0)

    def update_status(self, video_id: str, new_status: bool):
        """
        Params:
        - video_id: str
        - new_status: bool
        Returns:
        - update_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(video_id)},
                                              update={'$set': {'is_processed': new_status}})
        return (result.matched_count > 0)

    def delete_video(self, video_id: str) -> DeleteResult:
        """
        Params:
        - video_id: str
        Returns:
        - pymongo.results.DeleteResult
        """
        return self.__collection.delete_one(filter={'_id': ObjectId(video_id)})

    def delete_all_videos(self) -> DeleteResult:
        """
        Deletes all records,
        Returns:
        - pymongo.results.DeleteResult
        """
        return self.__collection.delete_many({})

    def get_all_video(self):
        return self.__collection.find({})
    
    def get_number_of_records(self):
        return self.__collection.count_documents({})
        