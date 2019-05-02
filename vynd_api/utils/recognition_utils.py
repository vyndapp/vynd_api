
import numpy as np

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