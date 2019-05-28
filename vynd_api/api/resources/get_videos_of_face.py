from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, FaceCollection

import json

class GetVideosOfFace(Resource):

   face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)

   def get(self):
      args = request.args
      if 'face_id' in args:
         face_id = args['face_id']
         try: 
            result = self.face_collection.get_videos_by_id(face_id)
            if len(result) == 0:
               return jsonify(error="id not found")
            return jsonify(video_ids=result[0]['video_ids'])
         except:
            return jsonify(error="invalid id")
      elif 'name' in args:
         name = args['name']
         result = self.face_collection.get_videos_by_name(name)
         for face in result:
            face['_id'] = str(face['_id'])
         return jsonify(video_ids=result)
      else: 
         return jsonify(error="")


