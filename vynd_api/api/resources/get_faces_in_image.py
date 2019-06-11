from flask_restful import Resource, request
from flask import jsonify
from ...search.image_search import search_by_image

from .. import FaceCollection

class GetFacesInImage(Resource):

    face_collection = FaceCollection()

    def get(self):
        image = request.args['image']
        faces_ids = search_by_image(image=image)
        return jsonify(videos=faces_ids)
