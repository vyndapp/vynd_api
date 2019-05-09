
import tensorflow as tf
import numpy as np

from typing import List, Optional, NamedTuple

from .image_face_recognizer import ImageFaceRecognizer
from .face_recognition_results import FaceRecognitionResults, FaceResult, FaceRecognitionStatus
from ..facedetection.face_detection_results import DetectedFace
from ..utils import vggface2_utlis, image_utils

class VGGFaceRecognizer(ImageFaceRecognizer):
    def __init__(self):
        vggface_path = '../models/vggface2/vggface2.pb' # frozen graph path
        self.default_dims = (160, 160) # input dimensions for the model
        self.sess = vggface2_utlis.load_model(vggface_path) # load the frozen model into a session object
        self.__initialize_tensors()

    def __initialize_tensors(self):
        input_tensor_name = "import/input:0"
        phase_train_tensor_name = "import/phase_train:0"
        embedding_tensor_name = "import/embeddings:0"
        self.input_tensor = self.sess.graph.get_tensor_by_name(input_tensor_name)
        self.phase_train_tensor = self.sess.graph.get_tensor_by_name(phase_train_tensor_name)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(embedding_tensor_name)

    def __del__(self): 
        """
        Closes tf.Session() object on object deletion
        """
        self.sess.close()

    def recognize(self, detected_faces: List[DetectedFace]) -> FaceRecognitionResults:
        """
        Creates a FaceRecognitionResults for each keyframe's list of DetectedFaces
        """
        face_results: List[FaceResult] = []
        for face in detected_faces:
            face_embedding = self.__image_to_embedding(face.image)
            face_results.append(FaceResult(features=face_embedding, 
                                           confidence=face.bbox.confidence,
                                           face_id=None))
        return FaceRecognitionResults(faces=face_results, status=FaceRecognitionStatus.SUCCESS)

    def __image_to_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Transform a cropped face's image into a vector of shape (512,) which is the feature vector
        """
        image = self.__preprocess_image(image)
        embedding = self.sess.run(self.embedding_tensor, 
                                  feed_dict={self.input_tensor: image, 
                                             self.phase_train_tensor: False})
        return np.squeeze(embedding)

    def __preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = image_utils.resize_image(image, new_shape=self.default_dims)
        image = vggface2_utlis.normalize_image(image)
        return np.array([image]) # single image (n, n, 3) to batch (1, n, n, 3)
