
import numpy as np

def euclidian_distance(embeddings1, embeddings2):
    """
        Computes the distance between two feature vectors
    """
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff))
    return dist
    
def cosine_similarity_distance(embeddings1, embeddings2):
    """
        Computes the distance between two feature vectors
    """
    dot = np.sum(np.multiply(embeddings1, embeddings2))
    norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
    similarity = min(dot / norm, 1.0)
    dist = np.arccos(similarity) / np.pi # arccos can only take values between -1 and 1 (inclusive)
    return dist