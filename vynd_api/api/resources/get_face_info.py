
from flask_restful import Resource, request
from flask import jsonify

from .. import CLIENT, FaceCollection
import numpy as np
import json

class GetFaces(Resource):

    face_collection = FaceCollection(CLIENT.vynd_db_test.face_collection)

    def get(self):
        faces = self.face_collection.get_faces_info()
        return jsonify(faces=faces)