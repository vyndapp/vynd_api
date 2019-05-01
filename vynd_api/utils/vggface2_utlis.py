
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

def distance(embeddings1, embeddings2, distance_metric=1):
    """
    Computes the distance between two feature vectors
    """
    if(distance_metric == 0):
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif(distance_metric == 1):
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = min(dot / norm, 1.0)
        dist = np.arccos(similarity) / np.pi # arccos can only take values between -1 and 1 (inclusive)
    return dist