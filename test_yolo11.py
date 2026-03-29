"""
YOLOv11 + OpenCV Complete Test Script
======================================
Supports: Webcam | Video File | Image File
Model is auto-downloaded on first run.

Install dependencies:
    pip install ultralytics opencv-python

Usage:
    python yolo11_opencv_test.py                  # webcam (default)
    python yolo11_opencv_test.py --source image.jpg
    python yolo11_opencv_test.py --source video.mp4
    python yolo11_opencv_test.py --model yolo11s.pt --conf 0.4
"""

import sys
import argparse
import time
import cv2
from pathlib import Path

# ── 1. Auto-install ultralytics if missing ─────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    print("[INFO] Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
    from ultralytics import YOLO


# ── 2. COCO class colors (80 classes) ─────────────────────────────────────────
import random
random.seed(42)
COLORS = {i: tuple(random.randint(50, 255) for _ in range(3)) for i in range(80)}


# ── 3. Draw detections on frame ────────────────────────────────────────────────
def draw_detections(frame, results, conf_threshold=0.25):
    """Draw bounding boxes and labels on the frame."""
    for result in results:
        boxes  = result.boxes
        names  = result.names

        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label  = names.get(cls_id, str(cls_id))
            color  = COLORS.get(cls_id, (0, 255, 0))

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            text       = f"{label} {conf:.2f}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness  = 2
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - baseline - 2),
                        font, font_scale, (0, 0, 0), thickness)

    return frame


# ── 4. Draw FPS overlay ────────────────────────────────────────────────────────
def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


# ── 5. Run on webcam / video ───────────────────────────────────────────────────
def run_video(model, source, conf, save_output):
    # source=0 → webcam, else file path
    cap_source = 0 if str(source) == "0" else str(source)
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_output:
        out_path = "yolo26_output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        print(f"[INFO] Saving output to: {out_path}")

    print("[INFO] Press 'q' to quit | 's' to save a snapshot")
    prev_time = time.time()
    snap_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        # Inference
        results = model(frame, conf=conf, verbose=False)

        # Annotate
        frame = draw_detections(frame, results, conf)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        frame = draw_fps(frame, fps)

        # Object count overlay
        total_objects = sum(len(r.boxes) for r in results if r.boxes is not None)
        cv2.putText(frame, f"Objects: {total_objects}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        cv2.imshow("YOLOv26 Detection", frame)

        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_name = f"snapshot_{snap_idx:03d}.jpg"
            cv2.imwrite(snap_name, frame)
            print(f"[INFO] Snapshot saved: {snap_name}")
            snap_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ── 6. Run on a single image ───────────────────────────────────────────────────
def run_image(model, source, conf):
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"[ERROR] Cannot read image: {source}")
        sys.exit(1)

    results = model(frame, conf=conf, verbose=False)
    frame   = draw_detections(frame, results, conf)

    # Print detected objects to console
    print("\n── Detected Objects ──────────────────")
    for result in results:
        names = result.names
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            print(f"  {names.get(cls_id, cls_id):20s}  conf={float(box.conf[0]):.3f}")
    print("─────────────────────────────────────\n")

    # Resize for display if image is too large
    h, w = frame.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    out_path = "yolo26_result.jpg"
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Result saved to: {out_path}")

    cv2.imshow("YOLOv26 Detection", frame)
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── 7. Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv26 + OpenCV Test")
    parser.add_argument("--model",  type=str, default="yolo11n.pt",
                        help="Model variant: yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt")
    parser.add_argument("--source", type=str, default="0",
                        help="Source: 0 for webcam, or path to image/video")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save",   action="store_true",
                        help="Save video output to yolo26_output.mp4")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  YOLOv11 OpenCV Test")
    print(f"{'='*50}")
    print(f"  Model  : {args.model}")
    print(f"  Source : {'Webcam' if args.source == '0' else args.source}")
    print(f"  Conf   : {args.conf}")
    print(f"{'='*50}\n")

    # Load model (auto-downloads weights on first run)
    print(f"[INFO] Loading model: {args.model}  (auto-downloads if not cached)")
    model = YOLO(args.model)
    print(f"[INFO] Model loaded successfully!\n")

    source = args.source
    is_image = Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if is_image:
        run_image(model, source, args.conf)
    else:
        run_video(model, source, args.conf, args.save)


if __name__ == "__main__":
    main()