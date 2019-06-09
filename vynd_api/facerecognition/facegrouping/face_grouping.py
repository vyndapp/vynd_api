from typing import List

from ..faceembedding.face_embedding import FaceEmbedding

def _get_face_connections(face_embeddings: List[FaceEmbedding], is_connected_fn):
    connection_matrix = [[False] * len(face_embeddings) for _ in range(len(face_embeddings))]

    for i, embedding_1 in enumerate(face_embeddings):
        for j in range(i + 1, len(face_embeddings)):
            embedding_2 = face_embeddings[j]
            connection_matrix[i][j] = connection_matrix[j][i] = is_connected_fn(embedding_1.features, \
                                      embedding_2.features)
    return connection_matrix

def group_faces(face_embeddings: List[FaceEmbedding], is_connected_fn):
    connection_matrix = _get_face_connections(face_embeddings, is_connected_fn)
    is_grouped = [False] * len(face_embeddings)

    def get_group(i):
        if is_grouped[i]:
            return []
        is_grouped[i] = True
        group = [face_embeddings[i]]
        for j in range(len(face_embeddings)):
            if connection_matrix[i][j]:
                group.extend(get_group(j))
        return group

    face_groups = []
    for i in range(len(face_embeddings)):
        cur_group = get_group(i)
        if cur_group:
            face_groups.append(cur_group)

    return face_groups
