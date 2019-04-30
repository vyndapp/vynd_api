from typing import Optional, List
from . import CLIENT
from ..entities.image import KeyFrame

import json

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