import numpy as np

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(embedding, db_embeddings, db_labels, threshold):
    best_dist = float("inf")
    best_label = "Unknown"

    for ref_emb, label in zip(db_embeddings, db_labels):
        dist = cosine_distance(embedding, ref_emb)

        if dist < best_dist:
            best_dist = dist
            best_label = label

    if best_dist > threshold:
        return "Unknown", best_dist

    return best_label, best_dist
