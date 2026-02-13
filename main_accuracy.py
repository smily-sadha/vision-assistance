import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from utils.face_utils import build_faiss_index, query_index

THRESHOLD = 0.6

# Load face database
print("[INFO] Loading face database...")
db = np.load("embeddings/face_db.npz")
db_embeddings = db["embeddings"].astype('float32')
db_labels = db["labels"]

print(f"Loaded labels: {set(db_labels)}")
print(f"Total embeddings: {len(db_embeddings)}")
print(f"Embedding dimension: {db_embeddings.shape[1]}")

# Build FAISS index
print("[INFO] Building FAISS index...")
index = build_faiss_index(db_embeddings)

# Load InsightFace
print("[INFO] Loading InsightFace (buffalo_l)...")
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use same size as when database was created
print("✓ InsightFace loaded successfully")

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1.0)

recognized_set = set()

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0

# Frame skipping for speed
PROCESS_EVERY_N_FRAMES = 2
frame_count = 0
last_faces = []

print("\n[INFO] Starting recognition...")
print("Press ESC to exit")
print("Press '+' to increase threshold (more lenient)")
print("Press '-' to decrease threshold (more strict)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame_count += 1
    
    # Process detection every Nth frame
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        try:
            # Detect and get embeddings using InsightFace
            faces = app.get(frame)
            last_faces = []
            
            for face in faces:
                try:
                    # Get bounding box
                    bbox = face.bbox.astype(int)
                    startX, startY, endX, endY = bbox
                    
                    # Ensure valid coordinates
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(frame.shape[1], endX)
                    endY = min(frame.shape[0], endY)
                    
                    # Get embedding from InsightFace
                    embedding = face.embedding
                    
                    # Check if embedding is valid
                    if embedding is None or len(embedding) == 0:
                        print("[WARNING] Invalid embedding received")
                        continue
                    
                    # Ensure embedding is the right shape
                    if len(embedding.shape) > 1:
                        embedding = embedding.flatten()
                    
                    # Check dimension matches database
                    if embedding.shape[0] != db_embeddings.shape[1]:
                        print(f"[ERROR] Embedding dimension mismatch!")
                        print(f"  Face embedding: {embedding.shape[0]}")
                        print(f"  Database expects: {db_embeddings.shape[1]}")
                        continue
                    
                    # Recognize using FAISS
                    label, dist = query_index(index, embedding, db_labels, THRESHOLD)
                    
                    # Convert distance to confidence
                    confidence = max(0, 1.0 - dist)
                    
                    # Store for display
                    last_faces.append({
                        'bbox': (startX, startY, endX, endY),
                        'label': label,
                        'confidence': confidence,
                        'distance': dist
                    })
                    
                    # Print recognition info
                    if label not in recognized_set and label != "Unknown":
                        print(f"[INFO] Recognized: {label} (dist: {dist:.3f}, conf: {confidence:.2%})")
                        recognized_set.add(label)
                
                except Exception as e:
                    print(f"[ERROR] Processing individual face: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        except Exception as e:
            print(f"[ERROR] During face detection: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Draw all faces (even on skipped frames)
    for face_info in last_faces:
        startX, startY, endX, endY = face_info['bbox']
        label = face_info['label']
        confidence = face_info['confidence']
        distance = face_info['distance']
        
        # Color based on confidence
        if label != "Unknown" and confidence > 0.5:
            color = (0, 255, 0)  # Green
        elif label != "Unknown":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Draw rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Draw label and confidence
        text = f"{label}: {confidence*100:.1f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw distance for debugging
        cv2.putText(frame, f"dist: {distance:.3f}", (startX, endY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Calculate FPS
    fps_frame_count += 1
    if fps_frame_count >= 30:
        fps_end_time = time.time()
        fps_display = fps_frame_count / (fps_end_time - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0
    
    # Display FPS and info
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Threshold: {THRESHOLD:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow("Face Recognition", frame)
    
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
print("\n[INFO] Recognition stopped")