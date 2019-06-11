from typing import List
from collections import deque

from bson import ObjectId

from .video_processing_results import VideoProcessingResult
from ..entities.keyframe import KeyFrame
from ..utils import image_utils
from ..facedetection.yolov3_detector import YOLOv3Detector
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.face_detection_results import FaceDetectionResults

from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.face_embedding import FaceEmbedding

from ..facerecognition.facematching.face_match_status import FaceMatchStatus
from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher, GroupMatch

from ..data.face_collection import FaceCollection
from ..data.keyframe_collection import KeyFrameCollection
from ..data.video_collection import VideoCollection
from ..data import CLIENT

class VideoProcessor:

    def __init__(self,
                 face_collection=CLIENT.vynd_db.face_collection,
                 keyframe_collection=CLIENT.vynd_db.keyframe_collection,
                 video_collection=CLIENT.vynd_db.video_collection):
        self.__face_collection = FaceCollection(face_collection)
        self.__keyframe_collection = KeyFrameCollection(keyframe_collection)
        self.__video_collection = VideoCollection(video_collection)
        self.__yolov3_image_face_detector: ImageFaceDetector = YOLOv3Detector()
        self.__image_face_embedder: ImageFacesEmbedder = VGGFaceEmbedder()
        self.__image_face_matcher: ImageFacesMatcher = ImageFacesMatcher(face_collection=face_collection)
        self.__default_face_dims = (100, 100)

    def is_invalid_id(self, video_id: str) -> bool:
        return not ObjectId.is_valid(video_id) or \
            self.__video_collection.get_video_by_id(video_id) is None

    def __insert_keyframes(self, video_id: str, keyframes: List[KeyFrame]) -> List[str]:
        return [self.__keyframe_collection.insert_new_keyframe(video_id=video_id)
                for keyframe in keyframes]

    def __add_keyframes_to_video(self, video_id: str, keyframe_ids: List[str]):
        for keyframe_id in keyframe_ids:
           self.__video_collection.add_keyframe(video_id=video_id,
                                                keyframe_id=keyframe_id)

    def process(self, video_id: str, keyframes: List[KeyFrame]) -> VideoProcessingResult:
        if self.is_invalid_id(video_id):
            return VideoProcessingResult.INVALID_VIDEO_ID

        # keyframe_ids = self.__insert_keyframes(video_id, keyframes)
        # self.__add_keyframes_to_video(video_id, keyframe_ids)

        face_embedding_results: List[FaceEmbedding] = []

        for keyframe in keyframes:
            # keyframe_id = self.__keyframe_collection.insert_new_keyframe(video_id=video_id)
            # self.__video_collection.add_keyframe(video_id=video_id,
            #                                      keyframe_id=keyframe_id)
            face_detection_result: FaceDetectionResults = self.__yolov3_image_face_detector.detect(keyframe)
            face_embedding_result: List[FaceEmbedding] = self.__image_face_embedder.faces_to_embeddings(face_detection_result)
            face_embedding_results.extend(face_embedding_result)

        group_matches: List[GroupMatch] = self.__image_face_matcher.match_faces(face_embedding_results)
        self.__update_db(video_id, group_matches)

        return VideoProcessingResult.SUCCESS

    def __update_db(self, video_id: str, group_matches: List[GroupMatch]):
        self.__add_new_faces(video_id, group_matches)
        self.__add_video_to_faces_assocs(video_id, group_matches)
        self.__add_faces_to_video_assocs(video_id, group_matches)
        self.__video_collection.update_status(video_id, True)

    def __add_new_faces(self, video_id, group_matches):
        new_faces = []
        for group_match in group_matches:
            if group_match.match_status == FaceMatchStatus.UNKNOWN_FACE:
                new_faces.append(self.__get_new_face(video_id, group_match))

        new_ids = deque(self.__face_collection.insert_new_faces(new_faces))
        for group_match in group_matches:
            if group_match.match_status == FaceMatchStatus.UNKNOWN_FACE:
                group_match.matched_id = new_ids.popleft()

    def __get_new_face(self, video_id, group_match) -> str:
        face_embedding = group_match.face_embeddings[0]
        resized_face_image = image_utils.resize_image(face_embedding.face_image, \
                new_shape=(self.__default_face_dims))
        return {
            'video_ids': [video_id],
            'features': face_embedding.features,
            'face_image': resized_face_image
        }

    def __add_video_to_faces_assocs(self, video_id: str, group_matches: List[GroupMatch]):
        face_ids = [group_match.matched_id for group_match in group_matches]
        self.__video_collection.add_faces(video_id, face_ids)

    def __add_faces_to_video_assocs(self, video_id: str, group_matches: List[GroupMatch]):
        face_ids = []
        for group_match in group_matches:
            if group_match.match_status == FaceMatchStatus.MATCHED:
                face_ids.append(group_match.matched_id)
        self.__face_collection.add_video_id_to_faces(face_ids, video_id)
