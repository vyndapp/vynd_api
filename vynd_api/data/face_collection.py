from bson.objectid import ObjectId
from pymongo.results import DeleteResult

import numpy as np

from . import CLIENT
from .db_utils import np_to_binary, binary_to_np, binary_to_b64

class FaceCollection:
    def __init__(self, collection=CLIENT.vynd_db.face_collection):
        self.__collection = collection

    def insert_new_face(self, keyframe_id: str, video_id: str, features: np.ndarray, face_image: np.ndarray) -> str:
        """
        Params:
        - keyframe_id: str
        - video_id: str
        - features: numpy.ndarray
        - face_image: numpy.ndarray
        Returns:
        - inserted_face_id: str
        """
        # TODO: change face_images -> face_image
        return str(self.__collection.insert_one(
            {
                'keyframe_ids': [keyframe_id],
                'video_ids': [video_id],
                'features': np_to_binary(features),
                'face_image': np_to_binary(face_image),
                'is_identified': False,
                'name': None
            }
        ).inserted_id)
    
    def insert_new_faces(self, faces):
        for face in faces:
            face['face_image'] = np_to_binary(face['face_image'])
            face['features'] = np_to_binary(face['features'])
            face['is_identified'] = False
            face['name'] = None
        if len(faces):
            inserted_ids = self.__collection.insert_many(faces).inserted_ids
            return list(map(str, inserted_ids))
        else:
            return []

    def get_face_by_id(self, face_id: str):
        """
        Params:
        - face_id: str
        Returns:
        - face_entity: dict
        """
        return self.__collection.find_one({"_id": ObjectId(face_id)})

    def get_videos_by_id(self, face_id: str):
        return list(self.__collection.find(filter={'_id': ObjectId(face_id)},
            projection={'video_ids': True, '_id': False}))

    def get_videos_by_name(self, name: str):
        return list(self.__collection.find(filter={'$text': {"$search": name}},
                                           projection={'video_ids': True, '_id': True}))

    def get_faces_info(self):
        faces = list(self.__collection.find(projection={'_id': True, 'name': True, 'face_image': True}))
        for face in faces:
            face['_id'] = str(face['_id'])
            face['face_image'] = binary_to_b64(face['face_image']).decode()
        return faces
    
    def add_keyframe_id(self, face_id: str, keyframe_id: str):
        """
        Params:
        - face_id: str
        - keyframe_id: str
        Returns:
        - insertion_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$push': {'keyframe_ids': keyframe_id}})
        return (result.matched_count > 0)

    def add_video_id(self, face_id: str, video_id: str):
        """
        Params:
        - face_id: str
        - video_id: str
        Returns:
        - insertion_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$push': {'video_ids': video_id}})
        return (result.matched_count > 0)
    
    def add_video_id_to_faces(self, face_ids: str, video_id: str):
        """
        Params:
        - face_id: str
        - video_id: str
        Returns:
        - insertion_result: bool
        """
        ids = [ObjectId(face_id) for face_id in face_ids]
        result = self.__collection.update_many(filter={'_id': {'$in': ids}},
                                              update={'$push': {'video_ids': video_id}})
        return (result.matched_count > 0)

    def update_name(self, face_id: str, name: str):
        """
        Params:
        - face_id: str
        - name: str
        Returns:
        - update_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'name': name}})
        return (result.matched_count > 0)

    def update_features(self, face_id: str, features: np.ndarray):
        """
        Params:
        - face_id: str
        - features: np.ndarray
        Returns:
        - update_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'features': np_to_binary(features)}})
        return (result.matched_count > 0)

    def update_face_image(self, face_id: str, face_image: np.ndarray):
        """
        Params:
        - face_id: str
        - face_image: np.ndarray
        Returns:
        - update_result: bool
        """
        result = self.__collection.update_one(filter={'_id': ObjectId(face_id)},
                                              update={'$set': {'face_image': np_to_binary(face_image)}})
        return (result.matched_count > 0)

    def delete_face(self, face_id: str) -> DeleteResult:
        """
        Params:
        - face_id: str
        Returns:
        - deletion_result: pymongo.results.DeleteResult
        """
        return self.__collection.delete_one(filter={'_id': ObjectId(face_id)})

    def get_all_faces_features(self):
        faces = list(self.__collection.find(projection={'features': True}))
        for face in faces:
            face['_id'] = str(face['_id'])
            face['features'] = binary_to_np(face['features'])
        return faces
    
    def get_all_faces_video_ids(self):
        return list(self.__collection.find(projection={'video_ids': True, '_id': False}))

    def delete_all_faces(self):
        """
        Deletes all records,
        Returns:
        - pymongo.results.DeleteResult
        """
        return self.__collection.delete_many({})

    def get_number_of_records(self):
        return self.__collection.count_documents({})