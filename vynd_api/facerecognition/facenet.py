import cv2
import os
import numpy as np

from pathlib import Path
from PIL import Image

from keras.models import model_from_json

from ..utils import image_utils, facenet_utils
from ..test.test_utils import get_all_cropped_images
from .image_face_recognizer import ImageFaceRecognizer

class FaceNetRecognizer(ImageFaceRecognizer):
    def __init__(self):
        facenet_path = 'facenet/nn4.small2.v1' # keras model's path
        self.default_dims = (96, 96) # default input dimensions to the model
        self.facenet = facenet_utils.load_model(facenet_path)

    def recognize(self, image: np.ndarray):
        return self.__image_to_embedding(image)

    def __image_to_embedding(self, image: np.ndarray, channels_first: bool=True) -> np.ndarray:
        """
        Transform a cropped face's image into a vector of shape (128,) which is the feature vector
        """
        image = self.__preprocess_image(image)
        embedding = self.facenet.predict_on_batch(image)[0]
        return np.squeeze(embedding)

    def __preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = image_utils.resize_image(image, self.default_dims)
        img = image[...,::-1]
        img = facenet_utils.normalize_image(img)
        return np.array([img])