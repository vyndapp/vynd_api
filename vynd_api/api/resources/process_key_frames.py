from typing import List

from flask_restful import Resource, request
from flask import jsonify

import numpy as np
from .. import base64_to_rgb
from .. import VideoProcessor

class ProcessKeyFrames(Resource):

   def post(self):
      video_id = request.form['video_id']
      key_frames = request.form['key_frames']
      
      key_frames: List[np.array] = map(base64_to_rgb, key_frames)
      result = VideoProcessor().process(video_id, key_frames)

      return jsonify(result=result)