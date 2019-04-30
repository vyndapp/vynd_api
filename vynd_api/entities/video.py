from typing import List
from .image import KeyFrame

import json

class Video:
    """
    video_id: str, 
    key_frames: List[KeyFrame], 
    is_processed: bool, 
    length_in_seconds: int,
    number_of_keyframes: int
    """
    __video_id: str
    __key_frames: List[KeyFrame]
    __is_processed: bool
    __length_in_seconds: int
    __number_of_keyframes: int
    # todo: add other video properties received from client
    
    def __init__(self, length_in_seconds: int, key_frames: List[KeyFrame], video_id: str = None):
        self.__video_id = video_id
        self.__length_in_seconds = length_in_seconds
        self.__key_frames = key_frames
        self.__number_of_keyframes = len(key_frames)
        self.__is_processed = False

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
    def length_in_seconds(self) -> int:
        return self.__length_in_seconds

    @property
    def number_of_keyframes(self) -> int:
        return len(self.__number_of_keyframes)

    @is_processed.setter
    def is_processed(self, value):
        self.__is_processed = value