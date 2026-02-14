class Detection:
    def __init__(self, bbox, confidence, class_id, label):
        self.bbox = bbox          # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.label = label
