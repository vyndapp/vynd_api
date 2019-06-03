import unittest
import os
import time
import shutil
from typing import List
from PIL import Image
import numpy as np

from .test_utils import save_img
from .. utils.recognition_utils import cosine_similarity_distance
from ..facedetection.face_detection_results import FaceDetectionResults, DetectedFace
from ..facedetection.face_detection_status import FaceDetectionStatus
from ..facedetection.bounding_box import BoundingBox
from ..facerecognition.faceembedding.face_embedding import FaceEmbedding
from .. facerecognition.faceembedding.vggface_embedder import VGGFaceEmbedder
from .. facerecognition.facegrouping.face_grouping import group_faces

def get_images():
   dir_name = os.path.dirname(os.path.realpath(__file__)) + '/resources/aligned_faces/'
   dir_ = os.fsencode(dir_name)
   images = []
   for file in os.listdir(dir_):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img = Image.open(dir_name + filename)
        np_img = np.asarray(img)
        images.append(np_img)
        img.close()
   return images


def save_results(groups):
    dir_name = os.path.dirname(os.path.realpath(__file__)) + '/resources/aligned_faces/grps/'
    if os.path.exists(dir_name):
       shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for i, g in enumerate(groups):
          for j, emb in enumerate(g):
             save_img(dir_name, emb.face_image, f'group{i}_{j}.png')

class TestFaceGrouping(unittest.TestCase):
    
    def setUp(self):
       images = get_images()
       bbox = BoundingBox(coordinates=[0, 0, 0, 0], confidence=0.9)
       detected_faces = list(map(lambda img: DetectedFace(image=img, bbox=bbox), images))
       detection_results = FaceDetectionResults("video_id", "keyframe_id", \
          FaceDetectionStatus.SUCCESS, detected_faces)
       face_embedder = VGGFaceEmbedder()
       self.face_embeddings: List[FaceEmbedding] = face_embedder.faces_to_embeddings(detection_results)

    def test_correct_grouping(self):
       start_time = time.time()
       grps: List[List[FaceEmbedding]] = group_faces(self.face_embeddings, \
         lambda features_a, features_b: cosine_similarity_distance(features_a, features_b) < 0.28)
       end_time = time.time()
       print(end_time - start_time)
       grp_sizes = list(map(len, grps))
       # save_results(grps)
       self.assertListEqual([2, 1, 1, 4, 3, 1, 1, 5, 2, 7, 2], grp_sizes)
        