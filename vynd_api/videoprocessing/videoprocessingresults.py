"""Success or Different types Errors for failing to process a video which could be repored to the client"""

from enum import Enum

class VideoProcessingResult(Enum):
    SUCCESS = 'success'
    FAILURE = 'failure'
    # todo: add specific types of failures which could be encountered during the processing
    # phase to be reported to api
