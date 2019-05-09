
import json

from . import CLIENT
from ..entities.keyframe import KeyFrame

class KeyFrameCollection:
    def __init__(self, collection=CLIENT.vynd_db.keyframe_collection):
        self._collection = collection

    def insert_keyframe(self, video_id: str, order_in_time: str, key_frame: KeyFrame) -> str:
        return str(self._collection.insert_one(
            {
                'video_id': video_id,
                'order_in_time': order_in_time
            }
        ).inserted_id)
    
    def delete_all_keyframes(self):
        self._collection.delete_many({})

    def get_all_keyframes(self):
        return self._collection.find()

# kfc = KeyFrameCollection()
# all_kf = kfc.get_all_keyframes()
# for x in all_kf:
#     print(x["video_id"])
# kfc.delete_all_keyframes()
