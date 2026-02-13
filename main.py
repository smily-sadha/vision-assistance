import cv2
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque
from utils.face_utils import build_faiss_index, query_index

THRESHOLD = 0.6
STABILITY_FRAMES = 10
UNKNOWN_TOLERANCE = 0.6   # Helps avoid flicker into Unknown

db = np.load("embeddings/face_db.npz")
db_embeddings = db["embeddings"].astype('float32')
db_labels = db["labels"]

print("Loaded labels:", set(db_labels))
print("Total embeddings:", len(db_embeddings))

index = build_faiss_index(db_embeddings)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

buffer = deque(maxlen=STABILITY_FRAMES)
last_stable_identity = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        label, dist = query_index(index,
                                  face.embedding,
                                  db_labels,
                                  THRESHOLD)

        buffer.append(label)

        if len(buffer) == STABILITY_FRAMES:
            stable_label = max(set(buffer), key=buffer.count)
        else:
            stable_label = label

        # ---- Unknown flicker suppression ----
        if stable_label == "Unknown" and last_stable_identity != "Unknown":
            stable_label = last_stable_identity

        last_stable_identity = stable_label

        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{stable_label} ({dist:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
