"""
Face Recognition Performance Evaluation
Generates confusion matrix, accuracy graphs, and performance metrics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from pathlib import Path
from insightface.model_zoo import get_model

# Configuration
DATASET_PATH = "faces"
DATABASE_PATH = "embeddings/face_db_ssd.npz"
OUTPUT_FOLDER = "evaluation_results"
CONF_THRESHOLD = 0.5


class FaceRecognitionEvaluator:
    """Evaluate face recognition performance"""
    
    def __init__(self):
        print("="*70)
        print("FACE RECOGNITION PERFORMANCE EVALUATOR")
        print("="*70)
        
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        print("\n[1/4] Loading models...")
        self._load_detector()
        self._load_embedder()
        
        print("\n[2/4] Loading face database...")
        self._load_database()
        
        print("\n[3/4] Collecting test data...")
        self.test_embeddings = []
        self.test_labels = []
        self.test_images = []
        self._collect_test_data()
        
        print("\n[4/4] Running evaluation...")
        self.predictions = []
        self.true_labels = []
        self.distances = []
        self._run_predictions()
    
    def _load_detector(self):
        prototxt = "model/deploy.prototxt"
        model = "model/res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, model)
        print("  ✓ SSD detector loaded")
    
    def _load_embedder(self):
        try:
            self.embedder = get_model('arcface_r100_v1')
            self.embedder.prepare(ctx_id=-1)
            print("  ✓ InsightFace embedder loaded")
        except:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self.embedder = app.models['recognition']
            print("  ✓ InsightFace buffalo_l loaded")
    
    def _load_database(self):
        db = np.load(DATABASE_PATH)
        self.db_embeddings = db["embeddings"].astype('float32')
        self.db_labels = db["labels"]
        self.unique_labels = sorted(set(self.db_labels))
        
        print(f"  ✓ Loaded {len(self.db_embeddings)} embeddings")
        print(f"  ✓ People: {self.unique_labels}")
    
    def _collect_test_data(self):
        person_folders = [f for f in Path(DATASET_PATH).iterdir() if f.is_dir()]
        
        for person_folder in person_folders:
            person_name = person_folder.name
            
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(list(person_folder.glob(f"*{ext}")))
            
            test_files = image_files[::3]
            
            print(f"  Testing {person_name}: {len(test_files)} images")
            
            for image_path in test_files:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                embedding = self._extract_embedding(image)
                if embedding is not None:
                    self.test_embeddings.append(embedding)
                    self.test_labels.append(person_name)
                    self.test_images.append(str(image_path))
        
        print(f"\n  ✓ Collected {len(self.test_embeddings)} test samples")
    
    def _extract_embedding(self, image):
        (h, w) = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > CONF_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face = image[y1:y2, x1:x2]
                if face.shape[0] < 20 or face.shape[1] < 20:
                    continue
                
                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (112, 112))
                    
                    embedding = self.embedder.get_feat(face_resized).flatten()
                    
                    # ✅ ONLY REAL FIX — NORMALIZATION
                    norm = np.linalg.norm(embedding)
                    print("Test embedding norm:", norm)
                    
                    if norm == 0:
                        return None
                    
                    embedding = embedding / norm
                    
                    return embedding
                    
                except:
                    continue
        
        return None
    
    def _run_predictions(self):
        threshold = 0.6
        
        preds = []
        dists = []
        
        for test_emb in self.test_embeddings:
            distances = np.linalg.norm(self.db_embeddings - test_emb, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist < threshold:
                predicted_label = self.db_labels[min_idx]
            else:
                predicted_label = "Unknown"
            
            preds.append(predicted_label)
            dists.append(min_dist)
        
        self.predictions = preds
        self.distances = dists
        self.true_labels = self.test_labels

    # ------------------ GRAPHS ------------------

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.true_labels, self.predictions, labels=self.unique_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=self.unique_labels,
                    yticklabels=self.unique_labels)
        
        plt.title("Confusion Matrix")
        plt.savefig(f"{OUTPUT_FOLDER}/confusion_matrix.png")
        plt.close()

    def plot_accuracy_by_threshold(self):
        thresholds = np.arange(0.3, 1.0, 0.05)
        accuracies = []

        for threshold in thresholds:
            preds = []

            for test_emb in self.test_embeddings:
                distances = np.linalg.norm(self.db_embeddings - test_emb, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]

                if min_dist < threshold:
                    predicted_label = self.db_labels[min_idx]
                else:
                    predicted_label = "Unknown"

                preds.append(predicted_label)

            accuracy = accuracy_score(self.test_labels, preds)
            accuracies.append(accuracy * 100)

        plt.figure()
        plt.plot(thresholds, accuracies)
        plt.axvline(x=0.6, linestyle='--')
        plt.title("Accuracy vs Threshold")
        plt.savefig(f"{OUTPUT_FOLDER}/accuracy_vs_threshold.png")
        plt.close()

    def plot_distance_distribution(self):
        plt.figure()
        plt.hist(self.distances, bins=30)
        plt.axvline(x=0.6, linestyle='--')
        plt.title("Distance Distribution")
        plt.savefig(f"{OUTPUT_FOLDER}/distance_distribution.png")
        plt.close()

    def plot_per_person_accuracy(self):
        person_accuracy = {}

        for person in self.unique_labels:
            indices = [i for i, label in enumerate(self.true_labels) if label == person]
            if indices:
                preds = [self.predictions[i] for i in indices]
                true = [self.true_labels[i] for i in indices]
                person_accuracy[person] = accuracy_score(true, preds) * 100

        plt.figure()
        plt.bar(person_accuracy.keys(), person_accuracy.values())
        plt.title("Per Person Accuracy")
        plt.savefig(f"{OUTPUT_FOLDER}/per_person_accuracy.png")
        plt.close()

    # ------------------ REPORT ------------------

    def generate_classification_report(self):
        report = classification_report(self.true_labels, self.predictions,
                                       labels=self.unique_labels, zero_division=0)

        with open(f"{OUTPUT_FOLDER}/classification_report.txt", "w") as f:
            f.write(report)

        print("\nClassification Report\n")
        print(report)

    def run_evaluation(self):
        self.plot_confusion_matrix()
        self.plot_accuracy_by_threshold()
        self.plot_distance_distribution()
        self.plot_per_person_accuracy()
        self.generate_classification_report()

        print("\nEvaluation complete")


if __name__ == "__main__":
    evaluator = FaceRecognitionEvaluator()
    evaluator.run_evaluation()
