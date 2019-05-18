
from typing import NamedTuple, List, Optional

from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults
from ..facerecognition.facematching.face_matching_results import FaceMatchingResults

class Features(NamedTuple):
    features: List[float]
    confidence: float

class Face():
    """
    Face:
    - face_id: str
    - keyframe_id: str
    - video_id: str
    - name: Optional[str]
    - feature_grp: List[Features]
    """

    __face_id: str
    __keyframe_id: str
    __video_id: str
    __name: Optional[str]
    __feature_grp: List[Features]

    def __init__(self, keyframe_id: str, video_id: str, features: Features, name=""):
        self.__keyframe_id = keyframe_id
        self.__video_id = video_id
        self.__name = name
        self.__feature_grp = [features]

    def add_feature_vector(self, feature_vector: Features):
        self.__feature_grp.append(feature_vector)

    @property
    def video_id(self) -> str:
        return self.__video_id
    
    @property
    def keyframe_id(self) -> str:
        return self.__keyframe_id
    
    @property
    def face_id(self) -> str:
        return self.__face_id

    @property
    def feature_grp(self) -> List[Features]:
        return self.__feature_grp

    @face_id.setter
    def face_id(self, value: str):
        self.__face_id = value