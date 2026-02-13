import cv2
import numpy as np
import time
from insightface.model_zoo import get_model
from utils.face_utils import build_faiss_index, query_index

THRESHOLD = 0.6
CONF_THRESHOLD = 0.5

# Load face database
db = np.load("embeddings/face_db.npz")
db_embeddings = db["embeddings"].astype('float32')
db_labels = db["labels"]

print("Loaded labels:", set(db_labels))
print("Total embeddings:", len(db_embeddings))

index = build_faiss_index(db_embeddings)

# ===== USE LIGHTWEIGHT SSD DETECTOR (fast) =====
print("[INFO] Loading lightweight face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# ===== USE INSIGHTFACE FOR EMBEDDING (accurate) =====
print("[INFO] Loading InsightFace embedding model...")
try:
    # Try to use the recognition model from InsightFace
    recognizer = get_model('arcface_r100_v1')
    recognizer.prepare(ctx_id=-1)  # Use CPU
    print("✓ Loaded ArcFace R100 model")
except:
    print("⚠ ArcFace R100 not available, trying buffalo_l...")
    try:
        recognizer = get_model('buffalo_l', root='~/.insightface/models/buffalo_l')
        recognizer.prepare(ctx_id=-1)
        print("✓ Loaded buffalo_l model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nFalling back to simple method...")
        recognizer = None

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

# Set camera to lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1.0)

recognized_set = set()

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    (h, w) = frame.shape[:2]

    # ===== LIGHTWEIGHT FACE DETECTION (same as 4_realtime.py) =====
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure valid coordinates
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Extract face
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # ===== EXTRACT EMBEDDING USING INSIGHTFACE =====
            try:
                if recognizer is not None:
                    # Prepare face for InsightFace
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (112, 112))
                    
                    # Get embedding
                    embedding = recognizer.get_feat(face_resized).flatten()
                else:
                    # Fallback: use simple face blob as embedding
                    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                     (0, 0, 0), swapRB=True, crop=False)
                    embedding = face_blob.flatten()
                
                # ===== RECOGNIZE USING FAISS =====
                label, dist = query_index(index, embedding, db_labels, THRESHOLD)
                
                # Convert distance to confidence
                proba = max(0, 1.0 - dist)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                label = "Unknown"
                proba = 0.0

            text = "{}: {:.2f}%".format(label, proba * 100)

            # Draw on frame
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Print once to terminal
            if label not in recognized_set and label != "Unknown":
                print(f"[INFO] Recognized: {label}")
                recognized_set.add(label)

    # Calculate FPS
    fps_frame_count += 1
    if fps_frame_count >= 30:
        fps_end_time = time.time()
        fps_display = fps_frame_count / (fps_end_time - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0

    # Display FPS
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to break
        break

cap.release()
cv2.destroyAllWindows()