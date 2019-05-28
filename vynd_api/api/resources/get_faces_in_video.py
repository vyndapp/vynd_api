from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, VideoCollection

class GetFacesInVideo(Resource):

   video_collection = VideoCollection(CLIENT.vynd_db_test.video_collection)

   def extractArgs(self):
      return request.args

   def get(self):
      video_id = self.extractArgs()['video_id']
      try:
         result = self.video_collection.get_faces(video_id)
         if len(result) == 0:
            return jsonify(error="id not found")
         return jsonify(faces_ids=result[0]['faces_ids'])
      except:
         return jsonify(error="invalid id")

