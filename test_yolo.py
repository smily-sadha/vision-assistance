import cv2
from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("assets/models/yolo/yolov8n.pt")  # adjust path if needed

print("✓ Model loaded")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("✓ Camera opened")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    annotated = results[0].plot()  # YOLO draws boxes

    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
