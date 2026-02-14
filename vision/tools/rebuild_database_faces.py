import cv2
import numpy as np
import os
import json
import hashlib
from pathlib import Path
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------

DATASET_PATH = "faces"
OUTPUT_PATH = "embeddings/face_db_ssd.npz"
TRACKER_PATH = "embeddings/processed_hashes.json"

CONF_THRESHOLD = 0.5
MIN_FACE_SIZE = 20

# ---------------- UTILS ----------------

def compute_file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---------------- INIT ----------------

print("=" * 60)
print("INCREMENTAL FACE DATABASE BUILDER (STABLE)")
print("=" * 60)

os.makedirs("embeddings", exist_ok=True)

processed_hashes = {}

if os.path.exists(TRACKER_PATH):
    with open(TRACKER_PATH, "r") as f:
        processed_hashes = json.load(f)
    print(f"✓ Loaded tracker: {len(processed_hashes)} entries")
else:
    print("✓ No tracker found — starting fresh")

existing_embeddings = []
existing_labels = []

if os.path.exists(OUTPUT_PATH):
    print("✓ Loading existing database...")
    db = np.load(OUTPUT_PATH)
    existing_embeddings = list(db["embeddings"])
    existing_labels = list(db["labels"])
    print(f"✓ Existing embeddings: {len(existing_embeddings)}")
else:
    print("✓ No database found — new DB will be created")

# Prevent tracker / DB mismatch
if processed_hashes and not os.path.exists(OUTPUT_PATH):
    print("⚠ Tracker exists but DB missing — resetting tracker")
    processed_hashes = {}

# ---------------- LOAD SSD DETECTOR ----------------

print("\n[INFO] Loading SSD face detector...")

prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("✓ SSD detector loaded")

# ---------------- LOAD INSIGHTFACE ----------------

print("[INFO] Loading InsightFace models (FaceAnalysis)...")

app = FaceAnalysis(name="buffalo_l")

app.prepare(ctx_id=-1)

recognizer = app.models.get("recognition", None)

if recognizer is None:
    raise RuntimeError("InsightFace recognition model failed to load")

print("✓ InsightFace recognition model ready")

# ---------------- DATASET CHECK ----------------

if not os.path.exists(DATASET_PATH):
    print(f"\n✗ ERROR: Dataset folder '{DATASET_PATH}' not found")
    exit(1)

person_folders = [p for p in Path(DATASET_PATH).iterdir() if p.is_dir()]

if not person_folders:
    print("\n✗ ERROR: No person folders found")
    exit(1)

print(f"\n✓ Found {len(person_folders)} identities")

# ---------------- PROCESSING ----------------

new_embeddings = []
new_labels = []

total_new = 0
total_skipped = 0
total_failed = 0

for person_dir in person_folders:

    label = person_dir.name

    print(f"\n{'=' * 60}")
    print(f"Processing: {label}")
    print("=" * 60)

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(person_dir.glob(ext))

    print(f"Images found: {len(image_paths)}")

    for img_path in image_paths:

        file_hash = compute_file_hash(img_path)

        if file_hash in processed_hashes:
            total_skipped += 1
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            print(f"  ✗ Cannot read: {img_path.name}")
            total_failed += 1
            continue

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        detector.setInput(blob)
        detections = detector.forward()

        faces = []

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > CONF_THRESHOLD:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = image[y1:y2, x1:x2]

                if face.shape[0] >= MIN_FACE_SIZE and face.shape[1] >= MIN_FACE_SIZE:
                    faces.append(face)

        # Enforce dataset correctness
        if len(faces) != 1:
            print(f"  ⚠ Skipping {img_path.name} (faces detected: {len(faces)})")
            total_failed += 1
            continue

        try:

            face = faces[0]

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (112, 112))

            embedding = recognizer.get_feat(face_resized).flatten()

            embedding = l2_normalize(embedding)   # CRITICAL

            new_embeddings.append(embedding)
            new_labels.append(label)

            processed_hashes[file_hash] = {
                "label": label,
                "file": img_path.name
            }

            total_new += 1

            print(f"  ✓ NEW: {img_path.name}")

        except Exception as e:

            print(f"  ✗ Embedding error: {img_path.name} → {e}")
            total_failed += 1

# ---------------- SAVE ----------------

all_embeddings = existing_embeddings + new_embeddings
all_labels = existing_labels + new_labels

if not all_embeddings:
    print("\n✗ ERROR: No embeddings available")
    exit(1)

print(f"\n{'=' * 60}")
print("SAVING DATABASE")
print("=" * 60)

np.savez(
    OUTPUT_PATH,
    embeddings=np.array(all_embeddings, dtype='float32'),
    labels=np.array(all_labels)
)

with open(TRACKER_PATH, "w") as f:
    json.dump(processed_hashes, f, indent=2)

print("\n✓ Database updated successfully")
print(f"Total embeddings: {len(all_embeddings)}")
print(f"New embeddings: {total_new}")
print(f"Skipped: {total_skipped}")
print(f"Failed: {total_failed}")

print("\nLabel distribution:")
unique_labels = sorted(set(all_labels))

for lbl in unique_labels:
    count = list(all_labels).count(lbl)
    print(f"  - {lbl}: {count}")

print("\n✓ Done. Run: python main.py")
