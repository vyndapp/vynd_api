
from flask_restful import Resource, request
from flask import jsonify, make_response
from flask_api import status

from .. import CLIENT, VideoCollection

class AddNewVideo(Resource):

    video_collection = VideoCollection()

    def post(self):
        if request.get_json() and request.get_json().get('extension'):
            extension = request.get_json().get('extension')
            video_id = self.video_collection.insert_new_video(video_extension=extension)
            request.stream.read()
            return make_response(jsonify(video_id=video_id), 
                                 status.HTTP_200_OK)
        else:
            return make_response(jsonify(error="Send video's extension along with new video addition's request"),
                                 status.HTTP_400_BAD_REQUEST)
