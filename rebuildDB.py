"""
This script rebuilds your face database using the SAME detection and embedding 
pipeline as the real-time recognition script, ensuring accuracy.

Your folder structure:
faces/
  ├── avan/
  ├── na/
  └── ne/
"""

import cv2
import numpy as np
import os
from pathlib import Path
from insightface.model_zoo import get_model

# Configuration - UPDATE THIS TO MATCH YOUR STRUCTURE
DATASET_PATH = "faces"  # Your faces folder
OUTPUT_PATH = "embeddings/face_db_ssd.npz"
CONF_THRESHOLD = 0.5

print("="*60)
print("FACE DATABASE BUILDER (SSD + InsightFace)")
print("="*60)

# Load SSD detector (same as real-time script)
print("\n[INFO] Loading SSD face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
print("✓ SSD detector loaded")

# Load InsightFace embedding model
print("[INFO] Loading InsightFace embedding model...")
try:
    recognizer = get_model('arcface_r100_v1')
    recognizer.prepare(ctx_id=-1)
    print("✓ Loaded ArcFace R100")
    use_insightface = True
except Exception as e:
    print(f"⚠ Could not load ArcFace R100: {e}")
    print("Trying to use buffalo_l recognition model...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        recognizer = app.models['recognition']
        print("✓ Loaded buffalo_l recognition model")
        use_insightface = True
    except Exception as e2:
        print(f"⚠ Could not load buffalo_l: {e2}")
        print("Using simple embedding extraction...")
        use_insightface = False

# Collect all face images
all_embeddings = []
all_labels = []

print(f"\n[INFO] Processing images from: {DATASET_PATH}")
print(f"[INFO] Looking for person folders...")

# Check if faces folder exists
if not os.path.exists(DATASET_PATH):
    print(f"\n✗ ERROR: Folder '{DATASET_PATH}' not found!")
    print(f"Please make sure the 'faces' folder exists in your project directory.")
    exit(1)

# Expected structure: faces/person_name/image.jpg
person_folders = [f for f in Path(DATASET_PATH).iterdir() if f.is_dir()]

if len(person_folders) == 0:
    print(f"\n✗ ERROR: No person folders found in '{DATASET_PATH}'")
    print("Expected structure:")
    print("  faces/")
    print("    ├── avan/")
    print("    ├── na/")
    print("    └── ne/")
    exit(1)

print(f"Found {len(person_folders)} person folders: {[f.name for f in person_folders]}")

total_processed = 0
total_failed = 0

for person_folder in person_folders:
    person_name = person_folder.name
    print(f"\n{'='*60}")
    print(f"Processing: {person_name}")
    print("="*60)
    
    image_count = 0
    failed_count = 0
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(list(person_folder.glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} images")
    
    for image_path in image_files:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ✗ Failed to load: {image_path.name}")
            failed_count += 1
            continue
        
        (h, w) = image.shape[:2]
        
        # Detect faces using SSD
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        detector.setInput(imageBlob)
        detections = detector.forward()
        
        # Process detected faces
        face_found = False
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
                
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                
                if fW < 20 or fH < 20:
                    continue
                
                # Extract embedding
                try:
                    if use_insightface:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (112, 112))
                        embedding = recognizer.get_feat(face_resized).flatten()
                    else:
                        # Fallback: simple blob embedding
                        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                         (0, 0, 0), swapRB=True, crop=False)
                        embedding = face_blob.flatten()
                    
                    all_embeddings.append(embedding)
                    all_labels.append(person_name)
                    image_count += 1
                    face_found = True
                    print(f"  ✓ {image_path.name}")
                    
                except Exception as e:
                    print(f"  ✗ Error extracting embedding from {image_path.name}: {e}")
                    failed_count += 1
                    continue
        
        if not face_found:
            print(f"  ⚠ No face detected in: {image_path.name}")
            failed_count += 1
    
    print(f"\nSummary for {person_name}:")
    print(f"  ✓ Successfully processed: {image_count}")
    print(f"  ✗ Failed: {failed_count}")
    
    total_processed += image_count
    total_failed += failed_count

# Save database
print(f"\n{'='*60}")
print("SAVING DATABASE")
print("="*60)

if len(all_embeddings) == 0:
    print("\n✗ ERROR: No embeddings were created!")
    print("Please check:")
    print("  1. Images exist in person folders")
    print("  2. Images contain visible faces")
    print("  3. Image files are valid (not corrupted)")
    exit(1)

print(f"Saving database to: {OUTPUT_PATH}")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

np.savez(OUTPUT_PATH,
         embeddings=np.array(all_embeddings),
         labels=np.array(all_labels))

print(f"\n{'='*60}")
print("DATABASE CREATED SUCCESSFULLY!")
print("="*60)
print(f"\nTotal embeddings: {len(all_embeddings)}")
print(f"Successfully processed: {total_processed}")
print(f"Failed: {total_failed}")
print(f"Unique people: {len(set(all_labels))}")
print(f"\nLabel distribution:")
for label in sorted(set(all_labels)):
    count = all_labels.count(label)
    print(f"  - {label}: {count} embeddings")

print(f"\n{'='*60}")
print("NEXT STEPS")
print("="*60)
print("\n1. Update your main script:")
print("   Change: db = np.load('embeddings/face_db_ssd.npz')")
print("\n2. Run recognition:")
print("   python main_ssd_insightface.py")
print("\nThis should now have HIGH accuracy and HIGH speed!")