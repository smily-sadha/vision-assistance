import cv2
import time
import numpy as np
import mediapipe as mp
from pathlib import Path

# -----------------------------
# USER INPUT
# -----------------------------

person_name = input("Enter person name: ").strip().lower()

if not person_name:
    raise ValueError("Invalid name")

SAVE_DIR = Path("faces") / person_name
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CONFIGURATION
# -----------------------------

MAX_IMAGES = 30
CAPTURE_INTERVAL = 0.5
BLUR_THRESHOLD = 80

# Bounding box expansion controls
SIDE_MARGIN = 0.25
TOP_MARGIN = 0.45        # Increased upward expansion (above hair)
BOTTOM_MARGIN = 0.55     # Keeps neck region

# -----------------------------
# MEDIAPIPE DETECTOR
# -----------------------------

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# -----------------------------
# BLUR CHECK
# -----------------------------

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD

# -----------------------------
# CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access camera")

print(f"\nCapturing face data for: {person_name}")
print("Move naturally. Press ESC to stop.\n")

saved = 0
last_capture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        detection = results.detections[0]

        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        box_w = x2 - x1
        box_h = y2 - y1

        # -----------------------------
        # EXPAND BOUNDING BOX
        # -----------------------------

        x1 = int(x1 - box_w * SIDE_MARGIN)
        x2 = int(x2 + box_w * SIDE_MARGIN)

        y1 = int(y1 - box_h * TOP_MARGIN)
        y2 = int(y2 + box_h * BOTTOM_MARGIN)

        # Clamp to frame safely
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size != 0:

            blurry = is_blurry(face_crop)
            status = "Blurry - Rejected" if blurry else "Clear"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            now = time.time()

            if not blurry and (now - last_capture_time) > CAPTURE_INTERVAL:
                img_path = SAVE_DIR / f"{saved:03d}.jpg"
                cv2.imwrite(str(img_path), face_crop)

                saved += 1
                last_capture_time = now

                print(f"Saved {saved}/{MAX_IMAGES}")

    cv2.imshow("Face Capture", frame)

    if saved >= MAX_IMAGES:
        print("\nCapture complete.\n")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        print("\nCapture stopped.\n")
        break

cap.release()
cv2.destroyAllWindows()
