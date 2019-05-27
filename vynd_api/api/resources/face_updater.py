
from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, FaceCollection

import numpy as np
import json

class FaceUpdater(Resource):

    face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)
    
    def post(self):
        face_id = request.form['face_id']
        name = request.form['name']

        update_result = self.face_collection.update_name(face_id, name)

        return jsonify(success=update_result)