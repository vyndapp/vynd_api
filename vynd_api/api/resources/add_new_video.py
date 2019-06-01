
from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, VideoCollection

class AddNewVideo(Resource):

    video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)

    def post(self):
        video_id = self.video_collection.insert_new_video()
        request.stream.read()
        return jsonify(video_id=video_id)
