from typing import Optional, List
from . import CLIENT
from ..entities.video import Video

import json

class VideoCollection:
    def __init__(self, collection=CLIENT.vynd_db.video_collection):
        self._collection = collection

    def insert_video(self, video: Video) -> str:
        return str(self._collection.insert_one(
            {
                'is_processed': video.is_processed,
                'length_in_seconds': video.length_in_seconds,
                'number_of_key_frames': len(video.key_frames)
            }
        ).inserted_id)

    def delete_all_videos(self):
        self._collection.delete_many({})

    def get_all_video(self):
        self._collection.find()

# vc = VideoCollection()
# vc.delete_all_videos()