
from flask import Blueprint
from flask_restful import Api

from .resources.face_updater import FaceUpdater
from .resources.get_face_info import GetFaces
from .resources.get_faces_in_video import GetFacesInVideo
from .resources.get_videos_of_face import GetVideosOfFace

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(FaceUpdater, '/update-face')
api.add_resource(GetFaces, '/get-faces')
api.add_resource(GetFacesInVideo, '/get-faces-in-video')
api.add_resource(GetVideosOfFace, '/get-videos-of-face')