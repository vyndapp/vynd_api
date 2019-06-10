
from flask_restful import Resource, request
from flask import jsonify, make_response
from flask_api import status

from .. import CLIENT, FaceCollection

import numpy as np
import json

class FaceUpdater(Resource):

    face_collection = FaceCollection()
    
    def post(self):
        if request.get_json() and request.get_json().get('face_id') and request.get_json().get('name'):
            face_id = request.get_json().get('face_id')
            name = request.get_json().get('name')

            update_result = self.face_collection.update_name(face_id, name)

            return jsonify(success=update_result)
        else:
            return make_response(jsonify(error="Send video's extension along with new video addition's request"),
                                 status.HTTP_400_BAD_REQUEST)