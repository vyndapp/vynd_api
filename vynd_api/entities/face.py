
from typing import NamedTuple, List, Optional

class Features(NamedTuple):
    features: List[float]
    confidence: float

class Face():
    _id: Optional[str]
    name: Optional[str]
    feature_grp: List[Features]
    video_ids: List[str]

    def __init__(self, feature_grp, video_ids, _id=None, name=None):
        self._id = _id
        self._name = name
        self.feature_grp = feature_grp
        self.video_ids = video_ids