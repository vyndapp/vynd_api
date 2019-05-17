from typing import Optional, List, NamedTuple
from . import CLIENT

from bson.objectid import ObjectId

import json

class FaceCollection:
    def __init__(self, collection=CLIENT.vynd_db_test.face_collection):
        self._collection = collection

    def insert_face(self, features: List[float], cosine_similarity_distance: float) -> str:
        return str(self._collection.insert_one(
            {
                'features': features,
                'cosine_similarity_distance': cosine_similarity_distance
            }
        ).inserted_id)

    def delete_all_faces(self):
        self._collection.delete_many({})

    def get_face_by_id(self, _id: str):
        return self._collection.find_one({"_id": ObjectId(_id)})

    def update_distance(self, _id: str, new_distance: float):
        self._collection.find_one_and_update(filter={"_id": _id}, 
                                             update={"$set": {"cosine_similarity_distance": new_distance}})

    def get_all_faces(self):
        return self._collection.find()

# fc = FaceCollection()
# fc.delete_all_faces()