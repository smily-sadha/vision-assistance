"""
INTEGRATED VISION ASSISTANT
Face Recognition + Emotion Detection + Motion Analysis + Distance + Voice AI
"""

import cv2
import time
import os
import json
import numpy as np
from dotenv import load_dotenv
from collections import deque

# Import existing components
from vision.recognition.recognition_manager import RecognitionManager

# Import new components
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("⚠ FER not installed. Run: pip install fer")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not installed. Run: pip install mediapipe")


class EmotionMotionAnalyzer:
    """Combined emotion and motion detection"""
    
    def __init__(self):
        """Initialize analyzers"""
        # Emotion detector
        if FER_AVAILABLE:
            try:
                self.emotion_detector = FER(mtcnn=False)
                print("  ✓ Emotion detector ready")
            except:
                self.emotion_detector = None
                print("  ✗ Emotion detector failed")
        else:
            self.emotion_detector = None
        
        # Motion detector
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.5
                )
                self.prev_landmarks = None
                self.motion_history = deque(maxlen=10)
                print("  ✓ Motion detector ready")
            except:
                self.pose = None
                print("  ✗ Motion detector failed")
        else:
            self.pose = None
    
    def detect_emotion(self, frame, face_bbox=None):
        """Detect emotion from face"""
        if self.emotion_detector is None:
            return "Neutral"
        
        try:
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                face_roi = frame[y1:y2, x1:x2]
            else:
                face_roi = frame
            
            result = self.emotion_detector.detect_emotions(face_roi)
            
            if result and len(result) > 0:
                emotions = result[0]['emotions']
                dominant = max(emotions.items(), key=lambda x: x[1])
                
                emotion_map = {
                    'happy': 'Happy',
                    'sad': 'Sad',
                    'angry': 'Angry',
                    'fear': 'Fear',
                    'surprise': 'Surprise',
                    'neutral': 'Neutral',
                    'disgust': 'Confused'
                }
                
                return emotion_map.get(dominant[0], 'Neutral')
            
            return "Neutral"
        except:
            return "Neutral"
    
    def detect_motion(self, frame):
        """Detect motion/activity"""
        if self.pose is None:
            return "Idle"
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Calculate movement
                movement = self._calculate_movement(landmarks)
                self.motion_history.append(movement)
                avg_movement = np.mean(self.motion_history) if self.motion_history else 0
                
                # Classify
                return self._classify_motion(landmarks, avg_movement)
            
            return "Idle"
        except:
            return "Idle"
    
    def _calculate_movement(self, landmarks):
        """Calculate movement between frames"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return 0.0
        
        movement = 0.0
        key_points = [0, 11, 12, 23, 24]
        
        for idx in key_points:
            if idx < len(landmarks):
                curr = landmarks[idx]
                prev = self.prev_landmarks[idx]
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                movement += np.sqrt(dx**2 + dy**2)
        
        self.prev_landmarks = landmarks
        return movement / len(key_points)
    
    def _classify_motion(self, landmarks, movement):
        """Classify motion type"""
        try:
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            knee_y = (left_knee.y + right_knee.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            
            # Sitting: knees higher than hips
            if knee_y < hip_y + 0.1:
                return "Sitting"
            
            # Movement-based
            if movement < 0.01:
                return "Standing"
            elif movement < 0.05:
                return "Walking"
            else:
                return "Running"
        except:
            return "Idle"


class IntegratedVisionAssistant:
    """Complete vision assistant with all features"""
    
    def __init__(self):
        print("="*70)
        print("INTEGRATED VISION ASSISTANT")
        print("Face + Emotion + Motion + Distance + Voice")
        print("="*70)
        
        load_dotenv()
        
        # Initialize components
        print("\n[1/4] Loading Face Recognition...")
        self.face_manager = self._init_face_recognition()
        
        print("\n[2/4] Loading Emotion & Motion Analysis...")
        self.analyzer = EmotionMotionAnalyzer()
        
        print("\n[3/4] Loading YOLO...")
        self.yolo_detector = self._init_yolo()
        
        print("\n[4/4] Loading Voice AI...")
        self.voice_assistant = self._init_voice()
        
        # Camera
        print("\nInitializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # State
        self.face_enabled = True
        self.emotion_enabled = True
        self.motion_enabled = True
        self.yolo_enabled = False
        self.voice_enabled = False
        
        # FPS
        self.fps_start = time.time()
        self.fps_count = 0
        self.fps_display = 0
        self.frame_counter = 0
        
        print("\n" + "="*70)
        print("✅ ALL SYSTEMS READY")
        print("="*70)
    
    def _init_face_recognition(self):
        """Initialize face recognition"""
        try:
            manager = RecognitionManager(
                database_path="embeddings/face_db_ssd.npz",
                embedder_method='insightface',
                threshold=0.6
            )
            print("  ✓ Face recognition ready")
            return manager
        except Exception as e:
            print(f"  ✗ Face recognition failed: {e}")
            return None
    
    def _init_yolo(self):
        """Initialize YOLO"""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("  ✓ YOLO ready")
            return model
        except:
            print("  ✗ YOLO not available")
            return None
    
    def _init_voice(self):
        """Initialize voice"""
        try:
            deepgram_key = os.getenv("DEEPGRAM_API_KEY")
            gemini_key = os.getenv("GEMINI_API_KEY")
            
            if not deepgram_key or not gemini_key:
                print("  ✗ Voice disabled - API keys missing")
                return None
            
            from vision.audio.tts.deepgram_tts import DeepgramTTS
            from vision.audio.llm.gemini_llm import GeminiLLM
            
            tts = DeepgramTTS(deepgram_key)
            llm = GeminiLLM(gemini_key, vision_impaired_mode=True)
            
            print("  ✓ Voice AI ready")
            return {'tts': tts, 'llm': llm}
        except:
            print("  ✗ Voice not available")
            return None
    
    def generate_context_response(self, emotion, motion, position):
        """Generate natural language response"""
        pos_text = {
            "Left": "on your left",
            "Right": "on your right",
            "Center": "in front of you"
        }
        
        pos = pos_text.get(position, "nearby")
        
        if motion == "Walking":
            return f"The person is walking {pos}"
        elif motion == "Running":
            return f"The person is running {pos}"
        elif motion == "Sitting":
            return f"The person is sitting {pos}"
        elif motion == "Standing":
            return f"The person is standing {pos}"
        else:
            return f"The person is {pos}"
    
    def get_position(self, bbox, frame_width):
        """Get spatial position"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        left_third = frame_width / 3
        right_third = 2 * frame_width / 3
        
        if center_x < left_third:
            return "Left"
        elif center_x > right_third:
            return "Right"
        else:
            return "Center"
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("KEYBOARD CONTROLS:")
        print("  E - Toggle Emotion Detection")
        print("  M - Toggle Motion Detection")
        print("  F - Toggle Face Recognition")
        print("  Y - Toggle YOLO")
        print("  V - Toggle Voice")
        print("  J - Print JSON Analysis")
        print("  SPACE - Snapshot")
        print("  ESC - Exit")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                display = frame.copy()
                h, w = display.shape[:2]
                self.frame_counter += 1
                
                # === FACE RECOGNITION ===
                face_results = []
                analysis_result = None
                
                if self.face_enabled and self.face_manager:
                    face_results = self.face_manager.recognize_frame(frame)
                    
                    # Process first face for emotion/motion
                    if len(face_results) > 0:
                        primary_face = face_results[0]
                        bbox = primary_face['bbox']
                        name = primary_face['name']
                        conf = primary_face['confidence']
                        
                        # === EMOTION DETECTION ===
                        emotion = "Neutral"
                        if self.emotion_enabled:
                            emotion = self.analyzer.detect_emotion(frame, bbox)
                        
                        # === MOTION DETECTION ===
                        motion = "Idle"
                        if self.motion_enabled:
                            motion = self.analyzer.detect_motion(frame)
                        
                        # === POSITION ===
                        position = self.get_position(bbox, w)
                        
                        # === CONTEXT RESPONSE ===
                        response = self.generate_context_response(emotion, motion, position)
                        
                        # Store analysis
                        analysis_result = {
                            "name": name,
                            "confidence": f"{conf*100:.0f}%",
                            "emotion": emotion,
                            "motion": motion,
                            "position": position,
                            "voice_response": response
                        }
                        
                        # Draw face box
                        x1, y1, x2, y2 = bbox
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        
                        # Label
                        text = f"{name}: {conf*100:.0f}%"
                        cv2.rectangle(display, (x1, y1-30), (x1+200, y1), color, -1)
                        cv2.putText(display, text, (x1, y1-8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                # === FPS ===
                self.fps_count += 1
                if self.fps_count >= 30:
                    fps_end = time.time()
                    self.fps_display = self.fps_count / (fps_end - self.fps_start)
                    self.fps_start = time.time()
                    self.fps_count = 0
                
                # === DISPLAY INFO ===
                y_pos = 30
                cv2.putText(display, f"FPS: {self.fps_display:.1f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                y_pos += 35
                
                # Status
                face_st = "ON" if self.face_enabled else "OFF"
                cv2.putText(display, f"Face: {face_st}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0,255,0) if self.face_enabled else (128,128,128), 2)
                y_pos += 30
                
                emotion_st = "ON" if self.emotion_enabled else "OFF"
                cv2.putText(display, f"Emotion: {emotion_st}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0,255,0) if self.emotion_enabled else (128,128,128), 2)
                y_pos += 30
                
                motion_st = "ON" if self.motion_enabled else "OFF"
                cv2.putText(display, f"Motion: {motion_st}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0,255,0) if self.motion_enabled else (128,128,128), 2)
                y_pos += 40
                
                # === ANALYSIS DISPLAY ===
                if analysis_result:
                    cv2.putText(display, f"Emotion: {analysis_result['emotion']}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0), 2)
                    y_pos += 35
                    
                    cv2.putText(display, f"Motion: {analysis_result['motion']}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,0), 2)
                    y_pos += 35
                    
                    cv2.putText(display, f"Position: {analysis_result['position']}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)
                    
                    # Response (bottom)
                    response = analysis_result['voice_response']
                    cv2.putText(display, response, (10, h-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                
                cv2.imshow("Integrated Vision Assistant", display)
                
                # === KEYS ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                
                elif key == ord('e') or key == ord('E'):
                    self.emotion_enabled = not self.emotion_enabled
                    print(f"Emotion: {'ON' if self.emotion_enabled else 'OFF'}")
                
                elif key == ord('m') or key == ord('M'):
                    self.motion_enabled = not self.motion_enabled
                    print(f"Motion: {'ON' if self.motion_enabled else 'OFF'}")
                
                elif key == ord('f') or key == ord('F'):
                    self.face_enabled = not self.face_enabled
                    print(f"Face: {'ON' if self.face_enabled else 'OFF'}")
                
                elif key == ord('y') or key == ord('Y'):
                    self.yolo_enabled = not self.yolo_enabled
                    print(f"YOLO: {'ON' if self.yolo_enabled else 'OFF'}")
                
                elif key == ord('v') or key == ord('V'):
                    self.voice_enabled = not self.voice_enabled
                    print(f"Voice: {'ON' if self.voice_enabled else 'OFF'}")
                
                elif key == ord('j') or key == ord('J'):
                    if analysis_result:
                        print("\n" + "="*70)
                        print(json.dumps(analysis_result, indent=2))
                        print("="*70)
                
                elif key == ord(' '):
                    filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 {filename}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Done!")


if __name__ == "__main__":
    app = IntegratedVisionAssistant()
    app.run()