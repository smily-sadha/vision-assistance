"""
REAL-TIME HUMAN ACTIVITY AND EMOTION ANALYSIS SYSTEM
Analyzes video stream for emotion, motion, position, and generates context-aware responses
"""

import cv2
import numpy as np
import json
import time
from collections import deque
#from fer import FER
import mediapipe as mp


class EmotionDetector:
    """Detect facial emotions using FER (Facial Emotion Recognition)"""
    
    def __init__(self):
        """Initialize emotion detector"""
        try:
            self.detector = FER(mtcnn=False)  # Faster without MTCNN
            print("✓ Emotion detector initialized")
        except Exception as e:
            print(f"✗ Emotion detector failed: {e}")
            self.detector = None
    
    def detect_emotion(self, frame, face_bbox=None):
        """
        Detect dominant emotion from frame
        
        Args:
            frame: Video frame (BGR)
            face_bbox: Optional (x1, y1, x2, y2) to crop face region
            
        Returns:
            emotion_label: str ("Happy", "Sad", "Angry", etc.)
        """
        if self.detector is None:
            return "Neutral"
        
        try:
            # Crop to face if bbox provided
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                face_roi = frame[y1:y2, x1:x2]
            else:
                face_roi = frame
            
            # Detect emotions
            result = self.detector.detect_emotions(face_roi)
            
            if result and len(result) > 0:
                emotions = result[0]['emotions']
                
                # Get dominant emotion
                dominant = max(emotions.items(), key=lambda x: x[1])
                emotion_label = dominant[0]
                
                # Map to allowed labels
                emotion_map = {
                    'happy': 'Happy',
                    'sad': 'Sad',
                    'angry': 'Angry',
                    'fear': 'Fear',
                    'surprise': 'Surprise',
                    'neutral': 'Neutral',
                    'disgust': 'Confused'  # Map disgust to confused
                }
                
                return emotion_map.get(emotion_label, 'Neutral')
            
            return "Neutral"
            
        except Exception as e:
            return "Neutral"


class MotionDetector:
    """Detect human motion/activity using pose estimation"""
    
    def __init__(self):
        """Initialize motion detector using MediaPipe Pose"""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Fastest model
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.prev_landmarks = None
            self.motion_history = deque(maxlen=10)  # Last 10 frames
            print("✓ Motion detector initialized")
        except Exception as e:
            print(f"✗ Motion detector failed: {e}")
            self.pose = None
    
    def calculate_movement(self, current_landmarks):
        """
        Calculate movement magnitude between frames
        
        Args:
            current_landmarks: MediaPipe pose landmarks
            
        Returns:
            movement_score: float (0 = still, >0.1 = moving)
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return 0.0
        
        # Calculate distance between key points
        movement = 0.0
        key_points = [0, 11, 12, 23, 24]  # Nose, shoulders, hips
        
        for idx in key_points:
            if idx < len(current_landmarks) and idx < len(self.prev_landmarks):
                curr = current_landmarks[idx]
                prev = self.prev_landmarks[idx]
                
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                movement += np.sqrt(dx**2 + dy**2)
        
        self.prev_landmarks = current_landmarks
        return movement / len(key_points)
    
    def classify_motion(self, landmarks, movement_score):
        """
        Classify motion type based on pose and movement
        
        Args:
            landmarks: MediaPipe pose landmarks
            movement_score: Movement magnitude
            
        Returns:
            motion_label: str ("Standing", "Walking", "Running", "Sitting", "Idle")
        """
        if landmarks is None or len(landmarks) == 0:
            return "Idle"
        
        # Get key landmarks
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            
            # Calculate torso angle (sitting vs standing)
            torso_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            
            # Sitting detection: knees higher than hips
            if knee_y < hip_y + 0.1:
                return "Sitting"
            
            # Movement-based classification
            if movement_score < 0.01:
                return "Standing"
            elif movement_score < 0.05:
                return "Walking"
            else:
                return "Running"
                
        except Exception as e:
            return "Idle"
    
    def detect_motion(self, frame):
        """
        Detect motion from frame
        
        Args:
            frame: Video frame (BGR)
            
        Returns:
            motion_label: str
        """
        if self.pose is None:
            return "Idle"
        
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Calculate movement
                movement = self.calculate_movement(landmarks)
                self.motion_history.append(movement)
                
                # Average movement over last frames
                avg_movement = np.mean(self.motion_history) if self.motion_history else 0
                
                # Classify motion
                motion = self.classify_motion(landmarks, avg_movement)
                return motion
            
            return "Idle"
            
        except Exception as e:
            return "Idle"


class PositionDetector:
    """Detect spatial position relative to camera"""
    
    @staticmethod
    def detect_position(bbox, frame_width):
        """
        Detect position (Left/Right/Center)
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            frame_width: Width of frame
            
        Returns:
            position_label: str ("Left", "Right", "Center")
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        # Divide frame into thirds
        left_boundary = frame_width / 3
        right_boundary = 2 * frame_width / 3
        
        if center_x < left_boundary:
            return "Left"
        elif center_x > right_boundary:
            return "Right"
        else:
            return "Center"


class ContextResponseGenerator:
    """Generate natural language responses based on motion + position"""
    
    @staticmethod
    def generate_response(emotion, motion, position):
        """
        Generate context-aware voice response
        
        Args:
            emotion: Emotion label
            motion: Motion label
            position: Position label
            
        Returns:
            response: str (natural language)
        """
        # Position mapping
        position_text = {
            "Left": "on your left",
            "Right": "on your right",
            "Center": "in front of you"
        }
        
        pos = position_text.get(position, "nearby")
        
        # Motion-based responses
        if motion == "Walking":
            return f"The person is walking {pos}"
        
        elif motion == "Running":
            if position == "Left":
                return f"The person is running towards your left"
            elif position == "Right":
                return f"The person is running towards your right"
            else:
                return f"The person is running {pos}"
        
        elif motion == "Sitting":
            if position == "Center":
                return f"The person is sitting {pos}"
            else:
                return f"The person is sitting slightly to your {position.lower()}"
        
        elif motion == "Standing":
            return f"The person is standing {pos}"
        
        else:  # Idle
            return f"The person is {pos}"


class HumanActivityAnalyzer:
    """
    Complete real-time human activity and emotion analysis system
    """
    
    def __init__(self):
        """Initialize all detectors"""
        print("="*70)
        print("HUMAN ACTIVITY & EMOTION ANALYSIS SYSTEM")
        print("="*70)
        
        print("\nInitializing components...")
        self.emotion_detector = EmotionDetector()
        self.motion_detector = MotionDetector()
        self.position_detector = PositionDetector()
        self.response_generator = ContextResponseGenerator()
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY")
        print("="*70)
    
    def analyze_frame(self, frame, face_bbox=None):
        """
        Analyze single frame for emotion, motion, position
        
        Args:
            frame: Video frame (BGR)
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            
        Returns:
            result: dict with emotion, motion, position, voice_response
        """
        h, w = frame.shape[:2]
        
        # Default bbox if not provided (full frame)
        if face_bbox is None:
            face_bbox = (0, 0, w, h)
        
        # Detect emotion
        emotion = self.emotion_detector.detect_emotion(frame, face_bbox)
        
        # Detect motion
        motion = self.motion_detector.detect_motion(frame)
        
        # Detect position
        position = self.position_detector.detect_position(face_bbox, w)
        
        # Generate response
        response = self.response_generator.generate_response(emotion, motion, position)
        
        # Structured output
        result = {
            "emotion": emotion,
            "motion": motion,
            "position": position,
            "voice_response": response
        }
        
        return result
    
    def analyze_frame_json(self, frame, face_bbox=None):
        """
        Analyze frame and return JSON string
        
        Args:
            frame: Video frame
            face_bbox: Optional face bbox
            
        Returns:
            json_string: JSON formatted result
        """
        result = self.analyze_frame(frame, face_bbox)
        return json.dumps(result, indent=2)


# ============================================================================
# DEMO APPLICATION
# ============================================================================

def demo_realtime_analysis():
    """Demo real-time analysis with webcam"""
    
    print("\n" + "="*70)
    print("REAL-TIME DEMO")
    print("Press ESC to exit")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = HumanActivityAnalyzer()
    
    # Optional: Load face detector for better face bbox
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except:
        face_cascade = None
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation
    fps_start = time.time()
    fps_count = 0
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Detect face (optional, for better emotion detection)
            face_bbox = None
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    # Get largest face
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    face_bbox = (x, y, x + fw, y + fh)
                    
                    # Draw face box
                    cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), 
                                (0, 255, 0), 2)
            
            # Analyze frame
            result = analyzer.analyze_frame(frame, face_bbox)
            
            # Calculate FPS
            fps_count += 1
            if fps_count >= 30:
                fps_end = time.time()
                fps_display = fps_count / (fps_end - fps_start)
                fps_start = time.time()
                fps_count = 0
            
            # Display results on frame
            y_pos = 30
            
            # FPS
            cv2.putText(display_frame, f"FPS: {fps_display:.1f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 40
            
            # Emotion
            cv2.putText(display_frame, f"Emotion: {result['emotion']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            y_pos += 35
            
            # Motion
            cv2.putText(display_frame, f"Motion: {result['motion']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)
            y_pos += 35
            
            # Position
            cv2.putText(display_frame, f"Position: {result['position']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            y_pos += 50
            
            # Voice Response (bottom)
            response_text = result['voice_response']
            # Wrap text if too long
            if len(response_text) > 50:
                words = response_text.split()
                line1 = ' '.join(words[:len(words)//2])
                line2 = ' '.join(words[len(words)//2:])
                cv2.putText(display_frame, line1, 
                           (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, line2, 
                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, response_text, 
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Human Activity & Emotion Analysis", display_frame)
            
            # Print JSON output
            if fps_count % 30 == 0:  # Every 30 frames
                print("\n" + "="*70)
                print(json.dumps(result, indent=2))
                print("="*70)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done!")


if __name__ == "__main__":
    demo_realtime_analysis()