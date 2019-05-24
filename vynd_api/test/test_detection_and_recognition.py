
import numpy as np

from typing import List

from ..facedetection.image_face_detector import ImageFaceDetector
from ..facedetection.faced import FacedDetector
from ..facedetection.video_face_detector import VideoFaceDetector
from ..facedetection.face_detection_results import FaceDetectionResults

from ..facerecognition.video_face_recognizer import VideoFaceRecognizer

from ..facerecognition.faceembedding.image_faces_embedder import ImageFacesEmbedder
from ..facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from ..facerecognition.faceembedding.face_embedding_results import FaceEmbeddingResults

from ..facerecognition.facematching.image_faces_matcher import ImageFacesMatcher
from ..facerecognition.facematching.face_matching_results import FaceMatchingResults

from ..entities.keyframe import KeyFrame
from ..entities.face import Face

from .test_utils import get_all_local_images, save_img

dummy_video_id = "dummy_video_id"
dummy_keyframe_id = "dummy_keyframe_id"

# detector construction
faced: ImageFaceDetector = FacedDetector(minimum_confidence = 0.9, offset_value = 20, pad_value=5)
video_faced: VideoFaceDetector = VideoFaceDetector(faced)

# get keyframes
key_frames: KeyFrame = []
local_images: List[np.ndarray] = get_all_local_images('resources/')
i = 1
for img in local_images:
    key_frame = KeyFrame(img, video_id=dummy_video_id)
    key_frame.keyframe_id = dummy_keyframe_id + str(i)
    i += 1
    key_frames.append(key_frame)

# detection phase
detection_results: List[FaceDetectionResults] = video_faced.get_detected_results(key_frames)

for i in range(len(detection_results)):
    if detection_results[i].detected_faces != None:
        for j in range(len(detection_results[i].detected_faces)):
            save_img('detected_faces/', detection_results[i].detected_faces[j].image, str(i) + str(j) + '.png')
print('done with detection')

# video recognizer construction
vgg_face = VGGFaceEmbedder()
video_recognizer = VideoFaceRecognizer(vgg_face)

matching_results = video_recognizer.recognize(detection_results)

for res in matching_results:
    for face in res.matched_faces:
        print(face.most_similar_face_name)
    # print(len(res.matched_faces))