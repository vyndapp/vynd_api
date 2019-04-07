from . import CLIENT

class VideoCollection:
    def __init__(self, collection=CLIENT.vynd_db.video_collection):
        self._collection = collection
        