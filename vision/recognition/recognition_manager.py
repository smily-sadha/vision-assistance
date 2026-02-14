"""
Recognition Manager - Coordinates face detection and recognition
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.face_utils import build_faiss_index, query_index


class RecognitionManager:
    """Manages complete face recognition pipeline"""
    
    def __init__(self, database_path="embeddings/face_db_ssd.npz",
                 detector_prototxt="model/deploy.prototxt",
                 detector_model="model/res10_300x300_ssd_iter_140000.caffemodel",
                 embedder_method='insightface',
                 threshold=0.6,
                 conf_threshold=0.5):
        """
        Initialize recognition manager
        
        Args:
            database_path: Path to face database (.npz)
            detector_prototxt: Path to SSD prototxt
            detector_model: Path to SSD model
            embedder_method: 'insightface' or 'openface'
            threshold: Recognition threshold
            conf_threshold: Detection confidence threshold
        """
        self.database_path = database_path
        self.threshold = threshold
        self.conf_threshold = conf_threshold
        self.embedder_method = embedder_method
        
        print("[INFO] Initializing Recognition Manager...")
        
        # Load database
        self._load_database()
        
        # Load detector
        self._load_detector(detector_prototxt, detector_model)
        
        # Load embedder
        self._load_embedder()
        
        print("✓ Recognition Manager ready")
    
    def _load_database(self):
        """Load face database"""
        try:
            if not os.path.exists(self.database_path):
                print(f"⚠ Database not found: {self.database_path}")
                self.embeddings = np.array([])
                self.labels = np.array([])
                self.index = None
                return
            
            db = np.load(self.database_path)
            self.embeddings = db["embeddings"].astype('float32')
            self.labels = db["labels"]
            
            print(f"✓ Loaded {len(self.embeddings)} embeddings")
            print(f"  People: {sorted(set(self.labels))}")
            
            # Build FAISS index
            self.index = build_faiss_index(self.embeddings)
            
        except Exception as e:
            print(f"✗ Error loading database: {e}")
            self.embeddings = np.array([])
            self.labels = np.array([])
            self.index = None
    
    def _load_detector(self, prototxt, model):
        """Load SSD face detector"""
        try:
            self.detector = cv2.dnn.readNetFromCaffe(prototxt, model)
            print("✓ SSD detector loaded")
        except Exception as e:
            print(f"✗ Error loading detector: {e}")
            raise
    
    def _load_embedder(self):
        """Load face embedder"""
        try:
            if self.embedder_method == 'insightface':
                # Try InsightFace
                try:
                    from insightface.model_zoo import get_model
                    self.embedder = get_model('arcface_r100_v1')
                    self.embedder.prepare(ctx_id=-1)
                    self.embedder_type = "arcface"
                    print("✓ InsightFace ArcFace loaded")
                except:
                    from insightface.app import FaceAnalysis
                    app = FaceAnalysis(name="buffalo_l")
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                    self.embedder = app.models['recognition']
                    self.embedder_type = "buffalo"
                    print("✓ InsightFace buffalo_l loaded")
            
            elif self.embedder_method == 'openface':
                self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
                self.embedder_type = "openface"
                print("✓ OpenFace loaded")
        
        except Exception as e:
            print(f"✗ Error loading embedder: {e}")
            raise
    
    def recognize_frame(self, frame):
        """
        Detect and recognize faces in frame
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of dicts: [{'bbox': (x1,y1,x2,y2), 'name': str, 
                            'confidence': float, 'distance': float}, ...]
        """
        results = []
        
        if self.index is None or len(self.embeddings) == 0:
            return results
        
        (h, w) = frame.shape[:2]
        
        # Detect faces
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        # Process each face
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure valid coordinates
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                
                if fW < 20 or fH < 20:
                    continue
                
                # Extract embedding
                try:
                    if self.embedder_type in ["arcface", "buffalo"]:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (112, 112))
                        embedding = self.embedder.get_feat(face_resized).flatten()
                    
                    elif self.embedder_type == "openface":
                        face_blob = cv2.dnn.blobFromImage(
                            face, 1.0 / 255, (96, 96),
                            (0, 0, 0), swapRB=True, crop=False
                        )
                        self.embedder.setInput(face_blob)
                        embedding = self.embedder.forward().flatten()
                    
                    # Recognize
                    label, dist = query_index(
                        self.index, 
                        embedding, 
                        self.labels, 
                        self.threshold
                    )
                    
                    confidence_score = max(0, 1.0 - dist)
                    
                    results.append({
                        'bbox': (startX, startY, endX, endY),
                        'name': label,
                        'confidence': confidence_score,
                        'distance': dist,
                        'detection_confidence': float(confidence)
                    })
                
                except Exception as e:
                    # If embedding fails, still add as Unknown
                    results.append({
                        'bbox': (startX, startY, endX, endY),
                        'name': "Unknown",
                        'confidence': 0.0,
                        'distance': 999.0,
                        'detection_confidence': float(confidence)
                    })
        
        return results
    
    def draw_results(self, frame, results):
        """
        Draw recognition results on frame
        
        Args:
            frame: Input frame
            results: List of results from recognize_frame()
            
        Returns:
            Frame with drawn results
        """
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            confidence = result['confidence']
            
            # Color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{name}: {confidence*100:.0f}%"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, text, (x1, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def update_threshold(self, new_threshold):
        """Update recognition threshold"""
        self.threshold = new_threshold
        print(f"Threshold updated to: {self.threshold:.2f}")