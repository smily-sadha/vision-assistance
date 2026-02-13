import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils.face_utils import find_best_match

THRESHOLD = 0.6

db = np.load("embeddings/face_db.npz")
db_embeddings = db["embeddings"]
db_labels = db["labels"]

print("Loaded labels:", set(db_labels))
print("Total embeddings:", len(db_embeddings))

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)   # FULL FRAME → Correct

    for face in faces:

        emb = face.embedding
        label, dist = find_best_match(
            emb, db_embeddings, db_labels, THRESHOLD
        )

        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{label} ({dist:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        print(f"Distance: {dist:.3f} → Predicted: {label}")

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
