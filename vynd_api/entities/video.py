
from typing import List, Set
from .keyframe import KeyFrame

class Video:
    """
    Video:
    - video_id: str
    - key_frames: List[KeyFrame]
    - is_processed: bool
    """
    __video_id: str
    __key_frames: List[KeyFrame]
    __key_frames_ids: List[str]
    __faces_ids: Set[str]
    __is_processed: bool
    # TODO: add other video properties received from client
    
    def __init__(self, key_frames: List[KeyFrame]):
        self.__key_frames = key_frames
        self.__key_frames_ids = []
        self.__faces_ids = set()
        self.__is_processed = False

    def add_keyframe(self, keyframe_id: str) -> None:
        self.__key_frames_ids.append(keyframe_id)

    def add_face(self, face_id: str) -> None:
        self.__faces_ids.add(face_id)

    @property
    def video_id(self) -> str:
        return self.__video_id

    @property
    def key_frames(self) -> List[KeyFrame]:
        return self.__key_frames

    @property
    def is_processed(self) -> bool:
        return self.__is_processed

    @property
    def key_frames_ids(self) -> List[str]:
        return self.__key_frames_ids 

    @property
    def faces_ids(self) -> Set[str]:
        return self.__faces_ids
    
    @video_id.setter
    def video_id(self, value: str):
        self.__video_id = value

    @is_processed.setter
    def is_processed(self, value: bool):
        self.__is_processed = value