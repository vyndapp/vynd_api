
from typing import Optional, List, NamedTuple
from bson.objectid import ObjectId

import numpy as np

# from ..facerecognition.facematching.face_match import FaceMatch
from . import CLIENT
from ..utils.numpy_encoder import NumpyEncoder
from .db_utils import np_to_binary

class FaceCollection:
    def __init__(self, collection=CLIENT.vynd_db_test.face_collection):
        self.__collection = collection

    def insert_new_face(self, keyframe_id: str, video_id: str, features: np.ndarray, face_image: np.ndarray, confidence: float) -> str:
        return str(self.__collection.insert_one(
            {
                'keyframe_ids': [keyframe_id],
                'video_ids': [video_id],
                'features': np_to_binary(features),
                'face_images': np_to_binary(face_image),
                'confidence_score': float(confidence),
                'is_identified': False,
                'name': 'unknown'
            }
        ).inserted_id)

    def get_face_by_id(self, _id: str):
        return self.__collection.find_one({"_id": ObjectId(_id)})
    
    def add_keyframe_id(self, face_id: str, keyframe_id: str):
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$push': {'keyframe_ids': keyframe_id}})
        return (result.matched_count > 0)
    
    def add_video_id(self, face_id: str, video_id: str):
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$addToSet': {'video_ids': video_id}})
        return (result.matched_count > 0)

    def update_confidence_score(self, face_id: str, confidence: float):
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'confidence_score': float(confidence)}})
        return (result.matched_count > 0)

    def update_name(self, face_id: str, name: str):
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'name': name}})
        return (result.matched_count > 0)

    def update_features(self, face_id: str, features: np.ndarray):
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'features': np_to_binary(features)}})
        return (result.matched_count > 0)

    def update_face_image(self, face_id: str, face_image: np.ndarray):
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'face_images': np_to_binary(face_image)}})
        return (result.matched_count > 0)

    def get_all_faces(self):
        return self.__collection.find()
    
    def delete_all_faces(self):
        self.__collection.delete_many({})

# fc = FaceCollection()
# fc.delete_all_faces()