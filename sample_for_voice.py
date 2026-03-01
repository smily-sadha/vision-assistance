"""
VISION ASSISTANT - Main Application
Face Recognition + YOLO Object Detection + Voice AI

Features:
- Real-time face recognition (SSD + InsightFace)
- YOLO object detection
- Voice AI (Deepgram TTS + Gemini)
- Auto-greeting for recognized people
- Context-aware AI responses

Controls:
- V: Toggle voice assistant
- F: Toggle face recognition
- Y: Toggle YOLO detection
- T: Test voice
- 1-5: Quick voice commands
- SPACE: Take snapshot
- ESC: Exit
"""

import cv2
import time
import os
from dotenv import load_dotenv
import numpy as np

# Import components
from vision.recognition.recognition_manager import RecognitionManager
from vision.audio.voice_assistant import VoiceAssistant
from vision.audio.stt.deepgram_stt import DeepgramSTT
from vision.audio.tts.deepgram_tts import DeepgramTTS
from vision.audio.llm.gemini_llm import GeminiLLM


class VisionAssistant:
    """Main Vision Assistant Application"""
    
    def __init__(self):
        print("="*70)
        print("VISION ASSISTANT")
        print("Face Recognition + YOLO + Voice AI")
        print("="*70)
        
        load_dotenv()
        
        # Initialize components
        print("\n[1/3] Loading Face Recognition...")
        self.face_manager = self._init_face_recognition()
        
        print("\n[2/3] Loading YOLO Detection...")
        self.yolo_detector = self._init_yolo()
        
        print("\n[3/3] Loading Voice AI...")
        self.voice_assistant = self._init_voice_assistant()
        
        # Camera
        print("\nInitializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # State
        self.face_enabled = True
        self.yolo_enabled = True
        self.voice_enabled = False
        
        self.recognized_people = set()
        self.current_people = []
        self.detected_objects = []
        self.last_greeting = {}
        self.greeting_cooldown = 30  # seconds
        
        # Performance
        self.fps_start = time.time()
        self.fps_count = 0
        self.fps_display = 0
        self.frame_counter = 0
        self.yolo_skip = 3  # Process YOLO every 3rd frame
        
        print("\n" + "="*70)
        print("✅ ALL SYSTEMS READY!")
        print("="*70)
    
    def _init_face_recognition(self):
        """Initialize face recognition system"""
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
        """Initialize YOLO object detector"""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("  ✓ YOLO object detection ready")
            return model
        except Exception as e:
            print(f"  ⚠ YOLO not available: {e}")
            return None
    
    def _init_voice_assistant(self):
        """Initialize voice AI system"""
        try:
            deepgram_key = os.getenv("DEEPGRAM_API_KEY")
            gemini_key = os.getenv("GEMINI_API_KEY")
            
            if not deepgram_key or not gemini_key:
                print("  ⚠ Voice disabled - API keys not set")
                return None
            
            stt = DeepgramSTT(deepgram_key)
            tts = DeepgramTTS(deepgram_key)
            llm = GeminiLLM(gemini_key)
            
            assistant = VoiceAssistant(stt, tts, llm)
            print("  ✓ Voice AI ready")
            return assistant
        except Exception as e:
            print(f"  ⚠ Voice not available: {e}")
            return None
    
    def detect_yolo_objects(self, frame):
        """Detect objects using YOLO"""
        if self.yolo_detector is None:
            return []
        
        try:
            results = self.yolo_detector(frame, conf=0.5, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_detector.names[cls_id]
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': cls_name,
                        'confidence': conf
                    })
            
            return detections
        except:
            return []
    
    def draw_face_results(self, frame, results):
        """Draw face recognition results on frame"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            conf = result['confidence']
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"{name}: {conf*100:.0f}%"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def draw_yolo_results(self, frame, detections):
        """Draw YOLO detection results on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            
            color = (255, 0, 0) if cls == 'person' else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"{cls}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def greet_person(self, name):
        """Greet a recognized person"""
        current_time = time.time()
        
        if name not in self.last_greeting or \
           current_time - self.last_greeting[name] > self.greeting_cooldown:
            
            self.last_greeting[name] = current_time
            
            if self.voice_assistant and self.voice_enabled:
                self.voice_assistant.greet_person(name)
            else:
                print(f"👋 Hello {name}!")
    
    def update_voice_context(self):
        """Update voice assistant with current vision context"""
        if self.voice_assistant and self.voice_enabled:
            self.voice_assistant.update_vision_context(
                recognized_people=self.current_people,
                face_count=len(self.current_people),
                objects=[d['class'] for d in self.detected_objects[:10]]
            )
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*70)
        print("KEYBOARD CONTROLS:")
        print("  V      - Toggle Voice Assistant ON/OFF")
        print("  T      - Test Voice (speak test message)")
        print("  F      - Toggle Face Recognition ON/OFF")
        print("  Y      - Toggle YOLO Detection ON/OFF")
        print("  SPACE  - Take Snapshot")
        print("  +/-    - Adjust face threshold")
        print("  ESC    - Exit")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                display_frame = frame.copy()
                self.frame_counter += 1
                
                # === FACE RECOGNITION ===
                face_results = []
                if self.face_enabled and self.face_manager:
                    face_results = self.face_manager.recognize_frame(frame)
                    self.current_people = [r['name'] for r in face_results if r['name'] != "Unknown"]
                    
                    # Greet new people
                    for result in face_results:
                        name = result['name']
                        if name != "Unknown" and name not in self.recognized_people:
                            self.recognized_people.add(name)
                            self.greet_person(name)
                    
                    display_frame = self.draw_face_results(display_frame, face_results)
                
                # === YOLO OBJECT DETECTION ===
                if self.yolo_enabled and self.yolo_detector:
                    if self.frame_counter % self.yolo_skip == 0:
                        self.detected_objects = self.detect_yolo_objects(frame)
                    display_frame = self.draw_yolo_results(display_frame, self.detected_objects)
                
                # === UPDATE VOICE CONTEXT ===
                self.update_voice_context()
                
                # === CALCULATE FPS ===
                self.fps_count += 1
                if self.fps_count >= 30:
                    fps_end = time.time()
                    self.fps_display = self.fps_count / (fps_end - self.fps_start)
                    self.fps_start = time.time()
                    self.fps_count = 0
                
                # === DISPLAY INFO ===
                y_pos = 30
                cv2.putText(display_frame, f"FPS: {self.fps_display:.1f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_pos += 35
                
                # Feature status
                face_status = "ON" if self.face_enabled else "OFF"
                face_color = (0, 255, 0) if self.face_enabled else (128, 128, 128)
                cv2.putText(display_frame, f"Face: {face_status}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
                y_pos += 30
                
                yolo_status = "ON" if self.yolo_enabled else "OFF"
                yolo_color = (0, 255, 0) if self.yolo_enabled else (128, 128, 128)
                cv2.putText(display_frame, f"YOLO: {yolo_status}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yolo_color, 2)
                y_pos += 30
                
                voice_status = "ON" if self.voice_enabled else "OFF"
                voice_color = (0, 255, 0) if self.voice_enabled else (128, 128, 128)
                cv2.putText(display_frame, f"Voice: {voice_status}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
                
                # Current people
                if self.current_people:
                    people_text = f"People: {', '.join(self.current_people)}"
                    cv2.putText(display_frame, people_text, 
                               (10, display_frame.shape[0] - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Object count
                if self.detected_objects:
                    obj_count = {}
                    for det in self.detected_objects:
                        cls = det['class']
                        obj_count[cls] = obj_count.get(cls, 0) + 1
                    
                    obj_text = ", ".join([f"{k}:{v}" for k, v in list(obj_count.items())[:5]])
                    cv2.putText(display_frame, obj_text, 
                               (10, display_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                cv2.imshow("Vision Assistant", display_frame)
                
                # === HANDLE KEYBOARD ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                
                elif key == ord('f') or key == ord('F'):
                    self.face_enabled = not self.face_enabled
                    print(f"Face Recognition: {'ON' if self.face_enabled else 'OFF'}")
                
                elif key == ord('y') or key == ord('Y'):
                    self.yolo_enabled = not self.yolo_enabled
                    print(f"YOLO Detection: {'ON' if self.yolo_enabled else 'OFF'}")
                
                elif key == ord('v') or key == ord('V'):
                    if self.voice_assistant:
                        self.voice_enabled = not self.voice_enabled
                        status = "ON" if self.voice_enabled else "OFF"
                        print(f"Voice Assistant: {status}")
                        if self.voice_enabled:
                            self.voice_assistant.speak("Voice assistant activated")
                    else:
                        print("⚠ Voice assistant not available")
                
                elif key == ord('t') or key == ord('T'):
                    if self.voice_assistant:
                        self.voice_assistant.speak("Voice system test successful")
                
                elif key == ord(' '):
                    filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Snapshot saved: {filename}")
                    if self.voice_assistant and self.voice_enabled:
                        self.voice_assistant.speak("Snapshot captured")
                
                elif key == ord('+') or key == ord('='):
                    if self.face_manager:
                        new_thresh = min(1.0, self.face_manager.threshold + 0.05)
                        self.face_manager.update_threshold(new_thresh)
                
                elif key == ord('-') or key == ord('_'):
                    if self.face_manager:
                        new_thresh = max(0.0, self.face_manager.threshold - 0.05)
                        self.face_manager.update_threshold(new_thresh)
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ Goodbye!")


if __name__ == "__main__":
    app = VisionAssistant()
    app.run()