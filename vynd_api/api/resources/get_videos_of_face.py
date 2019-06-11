from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, FaceCollection, VideoCollection

import json

class GetVideosOfFace(Resource):

   face_collection = FaceCollection()
   video_collection = VideoCollection()

   def get(self):
      args = request.args
      if 'face_id' in args:
         face_id = args['face_id']
         try: 
            result = self.face_collection.get_videos_by_id(face_id)
            if len(result) == 0:
               return jsonify(error="id not found")
            videos_ids = result[0]['video_ids']
            videos_extensions = self.video_collection.get_videos_extensions(videos_ids)
            videos_results = [{'video_id': video_id,
                               'extension': video_ext['extension']} for (video_id, video_ext) in zip(videos_ids, videos_extensions)]
            return jsonify(videos=videos_results)
         except:
            return jsonify(error="invalid id")
      elif 'name' in args:
         name = args['name']
         if(len(name) == 0):
            videos = self.video_collection.get_all_video()
            for video in videos:
               video['_id'] = str(video['_id'])
            return jsonify(videos=videos)
         result = self.face_collection.get_videos_by_name(name)
         
         face_videos_ids = [face['video_ids'] for face in result]
         unique_videos = set()
         for face_videos_list in face_videos_ids:
            for video_id in face_videos_list:
               unique_videos.add(video_id)
         unique_videos = list(unique_videos)
         videos_extensions = self.video_collection.get_videos_extensions(unique_videos)
         videos_results = [{'video_id': video_id,
                            'extension': video_ext['extension']} for (video_id, video_ext) in zip(unique_videos, videos_extensions)]
         
         return jsonify(videos=videos_results)
      else: 
         return jsonify(error="")


