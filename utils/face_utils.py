import numpy as np
import faiss

def build_faiss_index(embeddings):

    embeddings = embeddings.astype('float32')

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    return index


def query_index(index, embedding, labels, threshold):

    emb = np.expand_dims(embedding, axis=0).astype('float32')
    faiss.normalize_L2(emb)

    similarity, idx = index.search(emb, 1)

    sim = similarity[0][0]
    label = labels[idx[0][0]]

    dist = 1 - sim   # Convert similarity → cosine distance style

    if dist > threshold:
        return "Unknown", dist

    return label, dist
