from typing import List
from .image import KeyFrame

class Video:
    _video_id: str
    _key_frames: List[KeyFrame]
    _is_processed: bool
    # todo: add other video properties received from client
    
    def __init__(self, video_id: str, key_frames: List[KeyFrame]):
        self._video_id = video_id
        self._key_frames = key_frames
        self._is_processed = False

    @property
    def video_id(self) -> str:
        return self._video_id

    @property
    def key_frames(self) -> List[KeyFrame]:
        return self._key_frames

    @property
    def is_processed(self):
        return self._is_processed

    @is_processed.setter
    def is_processed(self, value):
        self._is_processed = value