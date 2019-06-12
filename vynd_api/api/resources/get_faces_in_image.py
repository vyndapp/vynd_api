from flask_restful import Resource, request
from flask import jsonify
from ...search.image_search import search_by_image

from .. import FaceCollection

class GetFacesInImage(Resource):

    face_collection = FaceCollection()

    # def get(self):
    #     image = request.args['image']
    #     faces_ids = search_by_image(image=image.encode())
    #     return jsonify(videos=faces_ids)
    
    def post(self):
        if request.get_json() and request.get_json().get('image'):
            image = request.get_json().get('image')
            faces_ids = search_by_image(image=image.encode())
            return jsonify(faces_ids=faces_ids)
        else:
            return jsonify(error="error")
