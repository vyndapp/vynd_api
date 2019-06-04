from typing import List
import numpy as np
import time

from .image_faces_embedder import ImageFacesEmbedder
from .face_embedding_results import FaceEmbeddingResults
from .face_embedding import FaceEmbedding
from .. import FaceDetectionResults
from .. import vggface2_utlis, image_utils

class VGGFaceEmbedder(ImageFacesEmbedder):
    """
    - Generates FaceEmbedding for each DetectedFace in a specific KeyFrame
    - Must be using 'with' clause
    """
    
    def __init__(self):
        self.__vggface_path = '../models/vggface2/vggface2.pb' # frozen graph path
        self.__default_dims = (160, 160) # input dimensions for the model
        self.__open_session()

    def __del__(self):
        self.__close_session()

    def faces_to_embeddings(self, detection_results: FaceDetectionResults) -> List[FaceEmbedding]:
        """
        Creates a FaceEmbeddingResults for each keyframe's list of DetectedFaces
        """
        face_embeddings: List[FaceEmbedding] = []
        for face in detection_results.detected_faces:
            face_embedding = self.image_to_embedding(face.image)
            face_embeddings.append(FaceEmbedding(features=face_embedding, 
                                                 keyframe_id=detection_results.keyframe_id,
                                                 video_id=detection_results.video_id,
                                                 face_image=face.image))

        return face_embeddings

    def __open_session(self):
        self.__sess = vggface2_utlis.load_model(self.__vggface_path) # load the frozen model into a session object
        self.__initialize_tensors()

    def __close_session(self):
        self.__sess.close()

    def __initialize_tensors(self):
        input_tensor_name = "import/input:0"
        phase_train_tensor_name = "import/phase_train:0"
        embedding_tensor_name = "import/embeddings:0"
        self.__input_tensor = self.__sess.graph.get_tensor_by_name(input_tensor_name)
        self.__phase_train_tensor = self.__sess.graph.get_tensor_by_name(phase_train_tensor_name)
        self.__embedding_tensor = self.__sess.graph.get_tensor_by_name(embedding_tensor_name)

    ## TODO: make embedder receive one image
    def image_to_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Transform a cropped face's image into a vector of shape (512,) which is the feature vector
        """
        image = self.__preprocess_image(image)
        embedding = self.__sess.run(self.__embedding_tensor, 
                                    feed_dict={self.__input_tensor: image, 
                                               self.__phase_train_tensor: False})
        return np.squeeze(embedding)

    def __preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        - Resize the image
        - Normalize the image
        """
        image = image_utils.resize_image(image, 
                                         new_shape=self.__default_dims)
        image = vggface2_utlis.normalize_image(image)
        return np.array([image]) # single image (n, n, 3) to batch (1, n, n, 3)
