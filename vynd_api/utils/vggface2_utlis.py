
import tensorflow as tf
import os
import numpy as np

from pathlib import Path

def load_model(graph_path: str) -> tf.Session():
    """
    Loads the frozen tensorflow model into a session
    """
    cur_dir = os.path.dirname(__file__)
    graph_path = str(Path(cur_dir, graph_path))
    sess = tf.Session()
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default() # set the loaded frozen graph as the session's default graph
        tf.import_graph_def(graph_def)
    return sess

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image so that the values become in range from -1 to 1 (inclusive)
    """
    return (image - 127.5) / 128.0

