
from flask_restful import Resource
from flask import jsonify

from .. import CLIENT, VideoCollection

class AddNewVideo(Resource):

    video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)

    def post(self):
        # video_id = self.video_collection.insert_new_video()
        return jsonify(video_id="123")