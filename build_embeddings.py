import cv2
import json
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

FACES_DIR = Path("faces")
DB_PATH = Path("embeddings/face_db.npz")
CACHE_PATH = Path("embeddings/processed_images.json")

DB_PATH.parent.mkdir(exist_ok=True)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------
# LOAD CACHE (processed images)
# -----------------------------

if CACHE_PATH.exists():
    with open(CACHE_PATH, "r") as f:
        processed = set(json.load(f))
    print("Loaded processed image cache.")
else:
    processed = set()
    print("Creating new cache.")

# -----------------------------
# LOAD EXISTING DATABASE
# -----------------------------

if DB_PATH.exists():
    db = np.load(DB_PATH)
    all_embeddings = list(db["embeddings"])
    all_labels = list(db["labels"])
    print("Loaded existing embedding database.")
else:
    all_embeddings = []
    all_labels = []
    print("Creating new embedding database.")

print("\nProcessing faces...\n")

new_count = 0

for person_dir in FACES_DIR.iterdir():

    if not person_dir.is_dir():
        continue

    print(f"Processing: {person_dir.name}")

    for img_path in person_dir.glob("*.*"):

        img_str = str(img_path)

        if img_str in processed:
            continue   # SKIP OLD IMAGES

        img = cv2.imread(img_str)
        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"  No face detected → {img_path.name}")
            continue

        emb = faces[0].embedding

        all_embeddings.append(emb)
        all_labels.append(person_dir.name)

        processed.add(img_str)
        new_count += 1

print(f"\nNew embeddings added: {new_count}")
print(f"Total embeddings stored: {len(all_embeddings)}")

# -----------------------------
# SAVE DATABASE
# -----------------------------

np.savez_compressed(
    DB_PATH,
    embeddings=np.array(all_embeddings),
    labels=np.array(all_labels)
)

# -----------------------------
# SAVE CACHE
# -----------------------------

with open(CACHE_PATH, "w") as f:
    json.dump(list(processed), f)

print("\nDatabase updated successfully.")
