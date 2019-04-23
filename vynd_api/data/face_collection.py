from typing import Optional, List, NamedTuple
import json
from data import CLIENT


## Will not necessarily be used but will act as current Schemas

class Features(NamedTuple):
    features: List[float]
    confidence: float

class Face():

    _id: Optional[str]
    name: Optional[str]
    feature_grp: List[Features]
    video_ids: List[str]

    def __init__(self, feature_grp, video_ids, _id=None, name=None):
        self._id = _id
        self._name = name
        self.feature_grp = feature_grp
        self.video_ids = video_ids

    def to_json(self): 
        return json.dumps(self.__dict__, indent=4)


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