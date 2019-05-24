
from typing import NamedTuple, List, Optional

from ..facerecognition.faceembedding.face_embedding import FaceEmbedding
from ..facerecognition.facematching.face_match import FaceMatch

class Face():
    """
    Face:
    - face_id: str
    - keyframe_id: str
    - video_id: str
    - face_embedding: FaceEmbedding
    - face_match: FaceMatch
    """

    __face_id: str
    __keyframe_id: str
    __video_id: str
    __name: Optional[str]
    __face_embedding: FaceEmbedding
    __face_match: FaceMatch

    def __init__(self, keyframe_id: str, face_embedding: FaceEmbedding, face_match: FaceMatch):
        self.__keyframe_id = keyframe_id
        self.__face_embedding = face_embedding
        self.__face_match = face_match

    @property
    def face_embedding(self):
        return self.__face_embedding
    
    @property
    def face_match(self):
        return self.__face_match

    @property
    def keyframe_id(self):
        return self.__keyframe_id
    
    @property
    def video_id(self):
        return self.__video_id

    @property
    def face_id(self):
        return self.__face_id
    
    @keyframe_id.setter
    def keyframe_id(self, value):
        self.__keyframe_id = value
    
    @video_id.setter
    def facevideo_id_id(self, value):
        self.__video_id = value

    @face_id.setter
    def face_id(self, value):
        self.__face_id = value