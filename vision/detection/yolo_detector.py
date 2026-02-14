from ultralytics import YOLO
import numpy as np
from core.types import Detection

class YoloDetector:

    def __init__(self, model_path="assets/models/yolo/yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf

    def detect(self, frame):
        """
        Input: BGR frame (numpy array)
        Output: List[Detection]
        """

        results = self.model(frame, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()

            for box, score, cls_id in zip(boxes, scores, class_ids):

                if score < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.astype(int)

                label = self.model.names[int(cls_id)]

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score),
                        class_id=int(cls_id),
                        label=label
                    )
                )

        return detections
