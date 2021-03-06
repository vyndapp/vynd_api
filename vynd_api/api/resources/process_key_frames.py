from typing import List
import json

from flask_restful import Resource, request
from flask_api import status
from flask import jsonify, make_response

from .. import KeyFrame
from .. import VideoProcessor, VideoProcessingResult
from .. import settings

class ProcessKeyFrames(Resource):

   def post(self):
      if request.get_json() and request.get_json().get('video_id') and request.get_json().get('base64_images'):
         video_id = request.get_json()['video_id']
         key_frames = request.get_json()['base64_images']

         request.stream.read()
         
         try:
            key_frames = [key_frame.encode() for key_frame in key_frames]
         except:
            return make_response(jsonify(error="you must send an array of base64 images of type List[decoded_base64]"), status.HTTP_406_NOT_ACCEPTABLE)

         key_frames: List[KeyFrame] = [KeyFrame(x) for x in key_frames]
         result: VideoProcessingResult = settings.video_processor.process(video_id, key_frames)

         if result == VideoProcessingResult.SUCCESS:
            return make_response(jsonify(result=str(result.SUCCESS)), status.HTTP_200_OK)
         return make_response(jsonify(result=str(result.INVALID_VIDEO_ID)), status.HTTP_400_BAD_REQUEST)
      else:
         request.stream.read()
         return make_response(jsonify(error="you must send (video_id: str) and (base64_images: List[str])"), status.HTTP_400_BAD_REQUEST)
