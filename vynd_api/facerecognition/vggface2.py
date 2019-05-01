
import tensorflow as tf
import numpy as np

from .image_face_recognizer import ImageFaceRecognizer
from ..utils.vggface2_utlis import load_model, normalize_image
from ..utils.image_utils import resize_image

class VGGFaceRecognizer(ImageFaceRecognizer):
    def __init__(self):
        vggface_path = '../models/vggface2/vggface2.pb' # frozen graph path
        self.default_dims = (160, 160) # input dimensions for the model

        input_tensor_name = "import/input:0"
        phase_train_tensor_name = "import/phase_train:0"
        embedding_tensor_name = "import/embeddings:0"
        self.sess = load_model(vggface_path) # load the frozen model into a session object
        self.input_tensor = self.sess.graph.get_tensor_by_name(input_tensor_name)
        self.phase_train_tensor = self.sess.graph.get_tensor_by_name(phase_train_tensor_name)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(embedding_tensor_name)

    def __del__(self): 
        """
        Closes tf.Session() object on object deletion
        """
        self.sess.close()

    def recognize(self, image: np.ndarray):
        return self.__image_to_embedding(image)

    def __image_to_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Transform a cropped face's image into a vector of shape (512,) which is the feature vector
        """
        image = resize_image(image, new_shape=self.default_dims)
        image = normalize_image(image)
        image = np.array([image])
        embedding = self.sess.run(self.embedding_tensor, feed_dict={self.input_tensor: image, self.phase_train_tensor: False})
        return np.squeeze(embedding)

# vgg = VGGFaceRecognizer()
# img = np.random.rand(256, 256, 3)
# emb = vgg.recognize(img)
# print(type(emb), len(emb))