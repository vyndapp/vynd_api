
import json

from . import CLIENT
from ..entities.keyframe import KeyFrame

from bson.objectid import ObjectId

class KeyFrameCollection:
    def __init__(self, collection=CLIENT.vynd_db_test.keyframe_collection):
        self.__collection = collection

    def insert_new_keyframe(self, key_frame: KeyFrame) -> str:
        return str(self.__collection.insert_one(
            {
                'video_id': key_frame.video_id,
                'timestamp': key_frame.timestamp,
                'faces_ids': []
            }
        ).inserted_id)

    def get_keyframe_by_id(self, keyframe_id: str):
        return self.__collection.find_one({'_id': ObjectId(keyframe_id)})

    def add_face(self, keyframe_id: str, face_id: str) -> bool:
        result = self.__collection.update_one(filter={'_id': ObjectId(keyframe_id)},
                                              update={'$addToSet': {'faces_ids': face_id}})
        return (result.matched_count > 0)                                            
    
    def delete_all_keyframes(self):
        self.__collection.delete_many({})

    def get_all_keyframes(self):
        return self.__collection.find()

# kfc = KeyFrameCollection()
# all_kf = kfc.get_all_keyframes()
# for x in all_kf:
#     print(x["video_id"])
# kfc.delete_all_keyframes()
