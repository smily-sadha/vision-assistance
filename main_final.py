import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils.face_utils import build_faiss_index, query_index

THRESHOLD = 0.6
CONF_THRESHOLD = 0.5

YOLO_CONF = 0.4
YOLO_SKIP_FRAMES = 2   # Run YOLO every N frames

print("=" * 70)
print("FACE RECOGNITION + YOLO OBJECT DETECTION (NO PERSON)")
print("=" * 70)

# ---------------- LOAD DATABASE ----------------

print("\n[INFO] Loading face database...")

try:
    db = np.load("embeddings/face_db_ssd.npz")
    print("✓ Using face_db_ssd.npz")
except:
    print("✗ No database found. Run rebuild_database_faces.py")
    exit(1)

db_embeddings = db["embeddings"].astype('float32')
db_labels = db["labels"]

print(f"  Total embeddings: {len(db_embeddings)}")
print(f"  People: {sorted(set(db_labels))}")
print(f"  Embedding dimension: {db_embeddings.shape[1]}")

# ---------------- FAISS INDEX ----------------

print("\n[INFO] Building FAISS index...")
index = build_faiss_index(db_embeddings)
print("✓ FAISS index ready")

# ---------------- FACE DETECTOR ----------------

print("\n[INFO] Loading SSD face detector...")

prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("✓ SSD detector loaded")

# ---------------- EMBEDDER ----------------

print("\n[INFO] Loading embedding model...")

db_dim = db_embeddings.shape[1]

embedder = None
embedder_type = "none"

if db_dim == 512:
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1)

        embedder = app.models['recognition']
        embedder_type = "insightface"

        print("✓ InsightFace embedder loaded")

    except Exception as e:
        print("✗ InsightFace load failed:", e)
        exit(1)

elif db_dim == 128:
    try:
        embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
        embedder_type = "openface"
        print("✓ OpenFace embedder loaded")

    except Exception as e:
        print("✗ OpenFace load failed:", e)
        exit(1)

# ---------------- YOLO ----------------

print("\n[INFO] Loading YOLO model...")
yolo_model = YOLO("assets/models/yolo/yolov8n.pt")
print("✓ YOLO loaded")

# ---------------- CAMERA ----------------

print("\n[INFO] Starting camera...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

time.sleep(1.0)

print("✓ Camera ready")

recognized_set = set()
frame_count = 0

fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0

print("\nRecognition active. ESC to exit.\n")

# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Frame grab failed")
        break

    frame_count += 1

    (h, w) = frame.shape[:2]

    # ---------------- YOLO OBJECT DETECTION ----------------

    if frame_count % YOLO_SKIP_FRAMES == 0:

        yolo_results = yolo_model(frame, conf=YOLO_CONF, verbose=False)

        if yolo_results and yolo_results[0].boxes is not None:

            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            scores = yolo_results[0].boxes.conf.cpu().numpy()
            class_ids = yolo_results[0].boxes.cls.cpu().numpy()

            for box, score, cls_id in zip(boxes, scores, class_ids):

                label = yolo_model.names[int(cls_id)]

                # 🚨 IMPORTANT: Ignore humans completely
                if label == "person":
                    continue

                x1, y1, x2, y2 = box.astype(int)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                cv2.putText(frame,
                            f"{label} {score:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 200, 0),
                            2)

    # ---------------- SSD FACE DETECTION ----------------

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < CONF_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w, endX), min(h, endY)

        face = frame[startY:endY, startX:endX]

        if face.shape[0] < 20 or face.shape[1] < 20:
            continue

        try:

            if embedder_type == "insightface":
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (112, 112))
                embedding = embedder.get_feat(face_resized).flatten()

            elif embedder_type == "openface":
                faceBlob = cv2.dnn.blobFromImage(
                    face,
                    1.0 / 255,
                    (96, 96),
                    (0, 0, 0),
                    swapRB=True,
                    crop=False
                )
                embedder.setInput(faceBlob)
                embedding = embedder.forward().flatten()

            else:
                continue

            label, dist = query_index(index, embedding, db_labels, THRESHOLD)

        except:
            label = "Unknown"
            dist = 1.0

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.putText(frame,
                    f"{label} ({dist:.2f})",
                    (startX, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

        if label not in recognized_set and label != "Unknown":
            print(f"✓ Recognized: {label} | Distance: {dist:.3f}")
            recognized_set.add(label)

    # ---------------- FPS ----------------

    fps_frame_count += 1

    if fps_frame_count >= 30:
        fps_end_time = time.time()
        fps_display = fps_frame_count / (fps_end_time - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0

    cv2.putText(frame,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2)

    cv2.imshow("Vision Assistant", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print("\nSession ended.")
