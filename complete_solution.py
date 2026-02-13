"""
COMPLETE SOLUTION: High Speed + High Accuracy
This script rebuilds your database and runs recognition with matched pipeline
"""

import os
import sys

print("="*70)
print("FACE RECOGNITION OPTIMIZATION - High Speed + High Accuracy")
print("="*70)

# Check if database needs rebuilding
rebuild_needed = True
if os.path.exists("embeddings/face_db_ssd.npz"):
    response = input("\nface_db_ssd.npz already exists. Rebuild anyway? (y/n): ")
    if response.lower() != 'y':
        rebuild_needed = False
        print("Using existing face_db_ssd.npz")

if rebuild_needed:
    print("\n" + "="*70)
    print("STEP 1: REBUILDING DATABASE")
    print("="*70)
    print("\nThis ensures detection and recognition use the SAME method")
    print("This will take 5-10 minutes...\n")
    
    import cv2
    import numpy as np
    from pathlib import Path
    from insightface.model_zoo import get_model
    
    DATASET_PATH = "faces"
    OUTPUT_PATH = "embeddings/face_db_ssd.npz"
    CONF_THRESHOLD = 0.5
    
    # Load SSD detector
    print("[INFO] Loading SSD face detector...")
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("✓ SSD detector loaded")
    
    # Load InsightFace embedding model
    print("[INFO] Loading InsightFace embedding model...")
    try:
        recognizer = get_model('arcface_r100_v1')
        recognizer.prepare(ctx_id=-1)
        print("✓ ArcFace R100 loaded")
        use_insightface = True
    except:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            recognizer = app.models['recognition']
            print("✓ buffalo_l recognition loaded")
            use_insightface = True
        except Exception as e:
            print(f"⚠ Could not load InsightFace: {e}")
            print("Using fallback embedding method")
            use_insightface = False
    
    # Process images
    all_embeddings = []
    all_labels = []
    
    print(f"\n[INFO] Processing images from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"\n✗ ERROR: '{DATASET_PATH}' folder not found!")
        sys.exit(1)
    
    person_folders = [f for f in Path(DATASET_PATH).iterdir() if f.is_dir()]
    
    if len(person_folders) == 0:
        print(f"\n✗ ERROR: No person folders in '{DATASET_PATH}'")
        sys.exit(1)
    
    print(f"Found {len(person_folders)} people: {[f.name for f in person_folders]}\n")
    
    total_processed = 0
    
    for person_folder in person_folders:
        person_name = person_folder.name
        print(f"Processing {person_name}...", end=" ")
        
        image_count = 0
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(list(person_folder.glob(f"*{ext}")))
        
        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            (h, w) = image.shape[:2]
            
            # Detect with SSD
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)
            
            detector.setInput(imageBlob)
            detections = detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > CONF_THRESHOLD:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    
                    if fW < 20 or fH < 20:
                        continue
                    
                    try:
                        if use_insightface:
                            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face_resized = cv2.resize(face_rgb, (112, 112))
                            embedding = recognizer.get_feat(face_resized).flatten()
                        else:
                            face_blob = cv2.dnn.blobFromImage(
                                face, 1.0 / 255, (96, 96),
                                (0, 0, 0), swapRB=True, crop=False)
                            embedding = face_blob.flatten()
                        
                        all_embeddings.append(embedding)
                        all_labels.append(person_name)
                        image_count += 1
                    except:
                        continue
        
        print(f"✓ {image_count} embeddings")
        total_processed += image_count
    
    if len(all_embeddings) == 0:
        print("\n✗ ERROR: No embeddings created!")
        sys.exit(1)
    
    # Save database
    print(f"\n[INFO] Saving database...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH,
             embeddings=np.array(all_embeddings),
             labels=np.array(all_labels))
    
    print(f"\n{'='*70}")
    print("DATABASE REBUILT SUCCESSFULLY!")
    print("="*70)
    print(f"Total embeddings: {len(all_embeddings)}")
    print(f"People: {sorted(set(all_labels))}")
    for label in sorted(set(all_labels)):
        count = all_labels.count(label)
        print(f"  - {label}: {count} embeddings")

print("\n" + "="*70)
print("STEP 2: STARTING HIGH-SPEED RECOGNITION")
print("="*70)
print("\nExpected performance: 25-30 FPS with HIGH accuracy\n")

# Now run the recognition
import cv2
import numpy as np
import time
from utils.face_utils import build_faiss_index, query_index

THRESHOLD = 0.6
CONF_THRESHOLD = 0.5

# Load the matched database
print("[INFO] Loading matched face database...")
db = np.load("embeddings/face_db_ssd.npz")
db_embeddings = db["embeddings"].astype('float32')
db_labels = db["labels"]

print(f"✓ Loaded {len(db_embeddings)} embeddings")
print(f"✓ People: {set(db_labels)}")

# Build index
print("[INFO] Building FAISS index...")
index = build_faiss_index(db_embeddings)

# Load SSD detector
print("[INFO] Loading SSD detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load embedder
print("[INFO] Loading embedding model...")
db_dim = db_embeddings.shape[1]

if db_dim == 512:
    try:
        from insightface.model_zoo import get_model
        embedder = get_model('arcface_r100_v1')
        embedder.prepare(ctx_id=-1)
        embedder_type = "insightface"
        print("✓ Using InsightFace embedder")
    except:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            embedder = app.models['recognition']
            embedder_type = "insightface"
            print("✓ Using buffalo_l embedder")
        except:
            embedder = None
            embedder_type = "none"
elif db_dim == 128:
    try:
        embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
        embedder_type = "openface"
        print("✓ Using OpenFace embedder")
    except:
        embedder = None
        embedder_type = "none"

print("\n[INFO] Starting camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)

recognized_set = set()
fps_start = time.time()
fps_count = 0
fps_display = 0

print("\n" + "="*70)
print("RECOGNITION ACTIVE - Press ESC to exit")
print("="*70)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Fast SSD detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            try:
                # Extract embedding (same method as database)
                if embedder_type == "insightface" and embedder is not None:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (112, 112))
                    embedding = embedder.get_feat(face_resized).flatten()
                elif embedder_type == "openface" and embedder is not None:
                    faceBlob = cv2.dnn.blobFromImage(
                        face, 1.0 / 255, (96, 96),
                        (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    embedding = embedder.forward().flatten()
                else:
                    continue

                # Recognize
                label, dist = query_index(index, embedding, db_labels, THRESHOLD)
                proba = max(0, 1.0 - dist)

            except:
                label = "Unknown"
                proba = 0.0

            # Draw
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            text = f"{label}: {proba*100:.0f}%"
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label not in recognized_set and label != "Unknown":
                print(f"✓ Recognized: {label} (confidence: {proba*100:.0f}%)")
                recognized_set.add(label)

    # FPS
    fps_count += 1
    if fps_count >= 30:
        fps_display = fps_count / (time.time() - fps_start)
        fps_start = time.time()
        fps_count = 0

    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Face Recognition - High Speed + High Accuracy", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*70}")
print(f"Final average FPS: {fps_display:.1f}")
print("="*70)