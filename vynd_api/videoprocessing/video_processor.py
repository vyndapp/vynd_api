from typing import List

import numpy as np
from bson import ObjectId

from .video_processing_results import VideoProcessingResult
from ..entities.video import Video
from ..entities.user import User
from ..entities.keyframe import KeyFrame
from ..facedetection.faced import FacedDetector
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.face_detection_results import FaceDetectionResults

from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults

from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher
from ..facerecognition.facematching.face_matching_results import FaceMatchingResults

from ..data.face_collection import FaceCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..data.video_collection import VideoCollection
from ..data import CLIENT

class VideoProcessor:
    """
    - video: Video
    - user: User
    - face_collection: Collection
    - keyframe_collection: Collection
    - video_collection: Collection
    """
    __video: Video
    __user: User

    def __init__(self,
                 face_collection=CLIENT.vynd_db_test.face_collection, 
                 keyframe_collection=CLIENT.vynd_db_test.keyframe_collection,
                 video_collection=CLIENT.vynd_db_test.video_collection):
        self.__face_collection = FaceCollection(face_collection)
        self.__keyframe_collection = KeyFrameCollection(keyframe_collection)
        self.__video_collection = VideoCollection(video_collection)
        self.__image_face_detector: ImageFaceDetector = FacedDetector()
        self.__image_face_embedder: ImageFacesEmbedder = VGGFaceEmbedder()
        self.__image_face_matcher: ImageFacesMatcher = ImageFacesMatcher(face_collection=face_collection)

    def is_invalid_id(self, video_id: str) -> bool:
        return not ObjectId.is_valid(video_id) or \
            self.__video_collection.get_video_by_id(video_id) is None

    def process(self, video_id: str, key_frames: List[KeyFrame]) -> VideoProcessingResult:
        # todo: Create a VideoDetector instance for a specific Image Detection Algo. Factory?
        """
        - Creates new entity for each keyframe found in video in DB
        - Matched faces found in keyframe with faces in DB
        - Insert faces that are not matched
        - Add all found faces in a Set() in video entity
        - Add all faces found inside a keyframe in a Set() in keyframe entity
        - Add keyframe_id in a Set() in face entity
        - Add video_id in a List() in face entity
        - Returns: VideoProcessingResult
        """

        if self.is_invalid_id(video_id):
            return VideoProcessingResult.INVALID_VIDEO_ID

        for keyframe in key_frames:
            keyframe_id = self.__keyframe_collection.insert_new_keyframe(video_id=video_id)
            self.__video_collection.add_keyframe(video_id=video_id,
                                                 keyframe_id=keyframe_id)
            keyframe.video_id = video_id
            keyframe.keyframe_id = keyframe_id
            face_detection_result: FaceDetectionResults = self.__image_face_detector.detect(keyframe)
            face_embedding_result: FaceEmbeddingResults = self.__image_face_embedder.faces_to_embeddings(face_detection_result)
            face_matching_results: FaceMatchingResults = self.__image_face_matcher.match_faces(face_embedding_result)
            
            for face in face_matching_results.matched_faces:
                self.__keyframe_collection.add_face(keyframe_id, face.face_id)
                self.__video_collection.add_face(video_id, face.face_id)
        self.__video_collection.update_status(video_id, True)

        return VideoProcessingResult.SUCCESS



# insert new video to DB
# get new inserted video ID 
# for each keyframe in video:
#     insert new keyframe to DB 
#     add new inserted keyframe ID to video entity
#     detect faces using image face detector
#     embbed faces using image face embedder
#     matches faces uing image face matcher
