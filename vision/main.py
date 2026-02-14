"""
FINAL OPTIMIZED VERSION
Requirements: Run rebuild_database_faces.py first to create face_db_ssd.npz
Performance: 25-30 FPS with HIGH accuracy
"""

import cv2
import numpy as np
import time
from utils.face_utils import build_faiss_index, query_index

THRESHOLD = 0.6
CONF_THRESHOLD = 0.5

print("="*70)
print("HIGH-SPEED + HIGH-ACCURACY FACE RECOGNITION")
print("="*70)

# Load matched database
print("\n[INFO] Loading face database...")
try:
    db = np.load("embeddings/face_db_ssd.npz")
    print("✓ Using matched database (face_db_ssd.npz)")
except:
    print("⚠ face_db_ssd.npz not found, trying face_db.npz...")
    try:
        db = np.load("embeddings/face_db.npz")
        print("⚠ Using original database - accuracy may be lower")
        print("  For best results, run: python rebuild_database_faces.py")
    except:
        print("✗ No database found!")
        print("  Please run: python rebuild_database_faces.py")
        exit(1)

db_embeddings = db["embeddings"].astype('float32')
db_labels = db["labels"]

print(f"  Total embeddings: {len(db_embeddings)}")
print(f"  People: {sorted(set(db_labels))}")
print(f"  Embedding dimension: {db_embeddings.shape[1]}")

# Build FAISS index
print("\n[INFO] Building FAISS index...")
index = build_faiss_index(db_embeddings)
print("✓ FAISS index ready")

# Load SSD detector (FAST)
print("\n[INFO] Loading SSD face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
print("✓ SSD detector loaded")

# Load embedder based on database dimension
print("\n[INFO] Loading embedding model...")
db_dim = db_embeddings.shape[1]

embedder = None
embedder_type = "none"

if db_dim == 512:
    # Try InsightFace
    try:
        from insightface.model_zoo import get_model
        embedder = get_model('arcface_r100_v1')
        embedder.prepare(ctx_id=-1)
        embedder_type = "insightface"
        print("✓ Using InsightFace ArcFace embedder (512D)")
    except:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            embedder = app.models['recognition']
            embedder_type = "insightface_buffalo"
            print("✓ Using InsightFace buffalo_l embedder (512D)")
        except Exception as e:
            print(f"✗ Could not load InsightFace embedder: {e}")
            print("  Recognition may not work properly")

elif db_dim == 128:
    # Try OpenFace
    try:
        embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
        embedder_type = "openface"
        print("✓ Using OpenFace embedder (128D)")
    except Exception as e:
        print(f"✗ Could not load OpenFace embedder: {e}")
        print("  Download: python download_all_models.py")

if embedder is None:
    print("\n✗ ERROR: No compatible embedder found!")
    print("  Database dimension:", db_dim)
    print("  Please rebuild database: python rebuild_database_faces.py")
    exit(1)

# Start camera
print("\n[INFO] Starting camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(1.0)
print("✓ Camera ready")

# Initialize
recognized_set = set()
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0

print("\n" + "="*70)
print("RECOGNITION ACTIVE")
print("="*70)
print("Expected FPS: 25-30")
print("Controls:")
print("  ESC - Exit")
print("  '+' - Increase threshold (more lenient)")
print("  '-' - Decrease threshold (more strict)")
print("="*70 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    (h, w) = frame.shape[:2]

    # SSD Face Detection (FAST - 300x300)
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    # Process each detected face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure valid coordinates
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Extract embedding (same method as database creation)
            try:
                if embedder_type == "insightface" or embedder_type == "insightface_buffalo":
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (112, 112))
                    embedding = embedder.get_feat(face_resized).flatten()
                
                elif embedder_type == "openface":
                    faceBlob = cv2.dnn.blobFromImage(
                        face, 1.0 / 255, (96, 96),
                        (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    embedding = embedder.forward().flatten()
                
                else:
                    continue

                # Recognize using FAISS
                label, dist = query_index(index, embedding, db_labels, THRESHOLD)
                confidence_score = max(0, 1.0 - dist)

            except Exception as e:
                label = "Unknown"
                confidence_score = 0.0

            # Draw result
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            text = f"{label}: {confidence_score*100:.0f}%"
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Print recognition (once per person)
            if label not in recognized_set and label != "Unknown":
                print(f"✓ Recognized: {label} (confidence: {confidence_score*100:.0f}%, distance: {dist:.3f})")
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Threshold: {THRESHOLD:.2f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('=') or key == ord('+'):
        THRESHOLD = min(1.0, THRESHOLD + 0.05)
        print(f"Threshold increased to: {THRESHOLD:.2f}")
    elif key == ord('-') or key == ord('_'):
        THRESHOLD = max(0.0, THRESHOLD - 0.05)
        print(f"Threshold decreased to: {THRESHOLD:.2f}")

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*70}")
print(f"Session ended - Average FPS: {fps_display:.1f}")
print("="*70)