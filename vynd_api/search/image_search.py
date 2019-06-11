import numpy as np
from ..entities.keyframe import KeyFrame
from ..facedetection.yolov3_detector import YOLOv3Detector
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher
from ..facerecognition.facematching.face_match_status import FaceMatchStatus
from ..data.face_collection import FaceCollection

def search_by_image(face_collection, image: np.ndarray):
    detection_results = YOLOv3Detector().detect(keyframe=KeyFrame(keyframe_image=image))
    embeddings = VGGFaceEmbedder().faces_to_embeddings(detection_results)
    matches = ImageFacesMatcher(face_collection=face_collection).match_faces(embeddings)
    faces_ids = []
    for match in matches:
        if match.match_status == FaceMatchStatus.MATCHED:
            faces_ids.append(match.matched_id)
    if not faces_ids:
        return []
    return FaceCollection(collection=face_collection).get_faces_videos(faces_ids=faces_ids)
