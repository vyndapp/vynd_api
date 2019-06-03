
import json

from . import CLIENT
from ..entities.keyframe import KeyFrame

from bson.objectid import ObjectId
from pymongo.results import DeleteResult

class KeyFrameCollection:
    def __init__(self, collection=CLIENT.vynd_db.keyframe_collection):
        self.__collection = collection

    def insert_new_keyframe(self, video_id: str, timestamp: int = 0) -> str:
        """
        Params:
        - video_id: str
        - timestamps: Optional[int]
        Returns:
        - inserted_keyframe_id: str
        """
        return str(self.__collection.insert_one(
            {
                'video_id': video_id,
                'timestamp': timestamp,
                'faces_ids': []
            }
        ).inserted_id)

    def get_keyframe_by_id(self, keyframe_id: str):
        """
        Params:
        - keyframe_id: str
        Returns:
        - keyframe_entity: dict
        """
        return self.__collection.find_one({'_id': ObjectId(keyframe_id)})

    def add_face(self, keyframe_id: str, face_id: str) -> bool:
        """
        Params:
        - keyframe_id: str
        - face_id: str
        Returns:
        - insertion_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(keyframe_id)},
                                              update={'$addToSet': {'faces_ids': face_id}})
        return (result.matched_count > 0)        

    def delete_keyframe(self, keyframe_id: str) -> DeleteResult:
        """
        Params:
        - keyframe_id: str
        Returns:
        - deletion_result: pymongo.results.DeleteResult
        """
        return self.__collection.delete_one(filter={'_id': ObjectId(keyframe_id)})
    
    def delete_all_keyframes(self):
        """
        Deletes all records,
        Returns:
        - pymongo.results.DeleteResult
        """
        return self.__collection.delete_many({})

    def get_all_keyframes(self):
        return self.__collection.find({})
    
    def get_number_of_records(self):
        return self.__collection.count_documents({})

# kfc = KeyFrameCollection()
# kfc.delete_all_keyframes()
