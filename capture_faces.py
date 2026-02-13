"""
SIMPLE AUTO FACE CAPTURE
Just enter name and it captures 30 images automatically
"""

import cv2
import os
import time
import numpy as np

# Settings
IMAGES_TO_CAPTURE = 30
FACES_FOLDER = "faces"
DELAY_BETWEEN_CAPTURES = 0.2  # Capture every 0.2 seconds

print("="*60)
print("AUTOMATIC FACE CAPTURE")
print("="*60)

# Only question: Get name
name = input("\nEnter name: ").strip()

if not name:
    print("Error: Name cannot be empty!")
    exit(1)

# Create folder
person_folder = os.path.join(FACES_FOLDER, name)
os.makedirs(person_folder, exist_ok=True)

# Count existing images
existing = len([f for f in os.listdir(person_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"\nFolder: {person_folder}")
if existing > 0:
    print(f"Existing images: {existing}")
    print(f"New images will start from #{existing + 1}")

# Load detector
print("\n[INFO] Loading face detector...")
try:
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("✓ Detector loaded")
except Exception as e:
    print(f"✗ Error loading detector: {e}")
    print("Make sure model files exist in 'model/' folder")
    exit(1)

# Start camera
print("[INFO] Starting camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2.0)

print("\n" + "="*60)
print("GET READY!")
print("="*60)
print("Position your face in the frame")
print("Camera will start capturing in 3 seconds...")
print("Move your head slightly for variety")
print("="*60)

# 3 second countdown
for i in range(3, 0, -1):
    ret, frame = cap.read()
    if ret:
        # Detect face for visual feedback
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(blob)
        detections = detector.forward()
        
        # Draw face box
        for j in range(detections.shape[2]):
            conf = detections[0, 0, j, 2]
            if conf > 0.5:
                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Countdown text
        cv2.putText(frame, f"Starting in {i}...", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("Face Capture", frame)
        cv2.waitKey(1)
    
    time.sleep(1)

# Start capturing
print("\n[INFO] CAPTURING...")

captured = 0
last_capture = 0

while captured < IMAGES_TO_CAPTURE:
    ret, frame = cap.read()
    if not ret:
        print("Error reading from camera")
        break
    
    current_time = time.time()
    (h, w) = frame.shape[:2]
    
    # Detect faces
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(blob)
    detections = detector.forward()
    
    face_saved = False
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure valid coordinates
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Draw box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Save if enough time passed
            if current_time - last_capture >= DELAY_BETWEEN_CAPTURES:
                # Extract and save face
                face = frame[startY:endY, startX:endX]
                
                if face.size > 0:  # Check if face is valid
                    img_num = existing + captured + 1
                    filename = f"{name}_{img_num:03d}.jpg"
                    filepath = os.path.join(person_folder, filename)
                    
                    cv2.imwrite(filepath, face)
                    captured += 1
                    last_capture = current_time
                    face_saved = True
                    
                    print(f"  [{captured}/{IMAGES_TO_CAPTURE}] {filename}")
            
            break  # Only capture first detected face
    
    # Display progress
    progress = int((captured / IMAGES_TO_CAPTURE) * 100)
    
    # Status text
    if face_saved:
        status = "CAPTURED!"
        color = (0, 255, 255)
    else:
        status = "Capturing..."
        color = (0, 255, 0)
    
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"{captured}/{IMAGES_TO_CAPTURE} ({progress}%)", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Progress bar
    bar_w = 500
    bar_h = 40
    cv2.rectangle(frame, (10, 90), (10 + bar_w, 90 + bar_h), (50, 50, 50), -1)
    filled = int((captured / IMAGES_TO_CAPTURE) * bar_w)
    cv2.rectangle(frame, (10, 90), (10 + filled, 90 + bar_h), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, 90), (10 + bar_w, 90 + bar_h), (255, 255, 255), 2)
    
    cv2.imshow("Face Capture", frame)
    
    # ESC to cancel
    if cv2.waitKey(1) & 0xFF == 27:
        print("\n[INFO] Cancelled by user")
        break

cap.release()
cv2.destroyAllWindows()

# Summary
print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"Name: {name}")
print(f"Images captured: {captured}")
print(f"Location: {person_folder}/")
print(f"Total for {name}: {existing + captured}")

if captured >= IMAGES_TO_CAPTURE:
    print("\n✓ SUCCESS! All images captured")
else:
    print(f"\n⚠ Only captured {captured}/{IMAGES_TO_CAPTURE}")

print("\nNext steps:")
print("  1. Add more people: python capture_faces_auto.py")
print("  2. Rebuild database: python rebuild_database_faces.py")
print("  3. Start recognition: python main_final.py")
print("="*60)