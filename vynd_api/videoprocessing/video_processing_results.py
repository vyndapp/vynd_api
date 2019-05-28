"""Success or Different types Errors for failing to process a video which could be repored to the client"""

from enum import Enum

class VideoProcessingResult(Enum):
    SUCCESS = 'success'
    FAILURE = 'failure'
    INVALID_VIDEO_ID = 'invalid_video_id'