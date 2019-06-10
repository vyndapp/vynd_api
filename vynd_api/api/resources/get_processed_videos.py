
from flask import jsonify
from flask_restful import Resource, request

from .. import CLIENT, VideoCollection

class GetProcessedVideos(Resource):

    video_collection = VideoCollection()

    def get(self):
        processed_videos = self.video_collection.get_processed_videos()
        for video in processed_videos:
            video['_id'] = str(video['_id'])
        processed_videos_results = [{'video_id': video['_id'],
                                     'extension': video['extension']} for video in processed_videos]
        return jsonify(processed_videos_results=processed_videos_results)
