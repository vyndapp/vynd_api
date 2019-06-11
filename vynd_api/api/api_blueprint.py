
from flask import Blueprint
from flask_restful import Api

from .resources.face_updater import FaceUpdater
from .resources.get_face_info import GetFaces
from .resources.get_faces_in_video import GetFacesInVideo
from .resources.get_videos_of_face import GetVideosOfFace
from .resources.add_new_video import AddNewVideo
from .resources.get_processed_videos import GetProcessedVideos
from .resources.process_key_frames import ProcessKeyFrames
from .resources.reset_db import ResetDB
from .resources.get_faces_in_image import GetFacesInImage

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(FaceUpdater, '/update-face')
api.add_resource(GetFaces, '/get-faces')
api.add_resource(GetFacesInVideo, '/get-faces-in-video')
api.add_resource(GetFacesInImage, '/get-faces-in-image')
api.add_resource(GetVideosOfFace, '/get-videos-of-face')
api.add_resource(AddNewVideo, '/add-new-video')
api.add_resource(GetProcessedVideos, '/get-processed-videos')
api.add_resource(ProcessKeyFrames, '/process-keyframes')
api.add_resource(ResetDB, '/reset-db')