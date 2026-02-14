yolo_detections = yolo_detector.detect(frame)

for det in yolo_detections:
    if det.label in IMPORTANT_CLASSES:
        event_bus.publish("OBJECT_DETECTED", det)
