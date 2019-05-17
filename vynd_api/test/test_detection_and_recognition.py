
import numpy as np

from typing import List

from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.faced import FacedDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..facedetection.face_detection_results import FaceDetectionResults

from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults

from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher
from ..facerecognition.facematching.face_matching_results import FaceMatchingResults

from ..entities.keyframe import KeyFrame

from .test_utils import get_all_local_images, save_img

# detector construction
faced: ImageFaceDetector = FacedDetector(minimum_confidence = 0.9, offset_value = 20)
video_faced: VideoFaceDetector = VideoFaceDetector(faced)

# get keyframes
key_frames: KeyFrame = []
local_images: List[np.ndarray] = get_all_local_images('resources/')
for img in local_images:
    key_frame = KeyFrame(img)
    key_frames.append(key_frame)

# detection phase
detection_results: List[FaceDetectionResults] = video_faced.get_detected_results(key_frames)
for i in range(len(detection_results)):
    if detection_results[i].detected_faces != None:
        for j in range(len(detection_results[i].detected_faces)):
            save_img('detected_faces/', detection_results[i].detected_faces[j].image, str(i) + str(j) + '.png')
print('done with detection')
# embedder construction
vggface: ImageFacesEmbedder = VGGFaceEmbedder()

# embedding phase
embedding_results: List[FaceEmbeddingResults] = list(map(
                                                    lambda detection_result: vggface.faces_to_embeddings(detection_result.detected_faces),
                                                    detection_results))
print('done with embedding')
# # matcher construction
# matcher = ImageFacesMatcher()

# # matching phase
# matching_results: List[FaceMatchingResults] = list(map(
#                                                     lambda embedding_result: matcher.match_faces(embedding_result.faces),
#                                                     embedding_results))
