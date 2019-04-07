from typing import List
from .image import KeyFrame

class Video:
    __video_id: str
    __key_frames: List[KeyFrame]
    __is_processed: bool
    # todo: add other video properties received from client
    
    def __init__(self, video_id: str, key_frames: List[KeyFrame]):
        self.__video_id = video_id
        self.__key_frames = key_frames
        self.__is_processed = False

    @property
    def video_id(self) -> str:
        return self.__video_id

    @property
    def key_frames(self) -> List[KeyFrame]:
        return self.__key_frames

    @property
    def is_processed(self):
        return self.__is_processed

    @is_processed.setter
    def is_processed(self, value):
        self.__is_processed = value