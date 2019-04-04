from .videoprocessingresults import VideoProcessingResult
from ..entities.video import Video
from ..entities.user import User

class VideoProcessor:
    _video: Video
    _user: User

    def __init__(self, video: Video, user: User):
        self._user = user
        self._video = video

    def process(self) -> VideoProcessingResult:
        # todo: Create a VideoDetector instance for a specific Image Detection Algo. Factory? 
        return VideoProcessingResult.SUCCESS