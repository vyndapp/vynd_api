
from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, FaceCollection, VideoCollection, KeyFrameCollection
import numpy as np
import json

class ResetDB(Resource):

    face_collection = FaceCollection()
    video_collection = VideoCollection()
    keyframe_collection = KeyFrameCollection()

    def post(self):
        self.face_collection.delete_all_faces()
        self.video_collection.delete_all_videos()
        self.keyframe_collection.delete_all_keyframes()
        return jsonify(success=True)