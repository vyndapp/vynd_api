from typing import Optional, List, NamedTuple
from . import CLIENT

import json

class FaceCollection:
    def __init__(self, collection=CLIENT.vynd_db.face_collection):
        self._collection = collection

    def insert_face(self, features, confidence, video_id) -> str:
        return str(self._collection.insert_one(
            {
                'feature_grp': [
                   {
                    'features': features,
                    'confidence': confidence
                   }
                ],
                'video_ids': [video_id]
            }
        ).inserted_id)
