from . import CLIENT

class FaceCollection:
    def __init__(self, collection=CLIENT.vynd_db.face_collection):
        self._collection = collection
    