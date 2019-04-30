import cv2
import os
import numpy as np

from pathlib import Path
from PIL import Image

from keras.models import model_from_json

from ..utils import face_recognition_utils, image_utils
from ..test.test_utils import get_all_cropped_images
from .image_face_recognizer import ImageFaceRecognizer

class FaceNetRecognizer(ImageFaceRecognizer):
    def __init__(self):
        facenet_path = 'facenet/nn4.small2.v1'
        self.default_dims = (96, 96)
        self.facenet = face_recognition_utils.load_model(facenet_path)

    def recognize(self, image: np.ndarray):
        return self.image_to_encoding(image)

    def image_to_encoding(self, image, channels_first=True):
        image = image_utils.resize_image(image, self.default_dims)
        img = image[...,::-1]
        img = image_utils.normalize_image(img, channels_first=channels_first)
        x_train = np.array([img])
        embedding = self.facenet.predict_on_batch(x_train)[0]
        return embedding
