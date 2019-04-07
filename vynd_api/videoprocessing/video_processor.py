from typing import List

from .video_processing_results import VideoProcessingResult
from ..entities.video import Video
from ..entities.user import User
from ..facedetection.faced import FacedDetector
from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..facedetection.face_detection_results import FaceDetectionResults

class VideoProcessor:
    _video: Video
    _user: User

    def __init__(self, video: Video, user: User):
        self.__user = user
        self.__video = video

    def process(self) -> VideoProcessingResult:
        # todo: Create a VideoDetector instance for a specific Image Detection Algo. Factory? 
        faced: ImageFaceDetector = FacedDetector()
        detector: VideoFaceDetector = VideoFaceDetector(faced)
        detection_results: List[FaceDetectionResults] = detector.get_detected_results(key_frames = self.__video.key_frames)
        return VideoProcessingResult.SUCCESS