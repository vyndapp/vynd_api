
from flask import Blueprint
from flask_restful import Api

from .resources.face_updater import FaceUpdater
from .resources.get_face_info import GetFaceInfo

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(FaceUpdater, '/update-face')
api.add_resource(GetFaceInfo, '/get-faces')