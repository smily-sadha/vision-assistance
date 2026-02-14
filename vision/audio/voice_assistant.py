"""
Voice Assistant Coordinator
Manages STT, TTS, and LLM integration with face recognition
"""

import asyncio
import threading
from queue import Queue


class VoiceAssistant:
    """AI Voice Assistant with vision capabilities"""
    
    def __init__(self, stt, tts, llm):
        """
        Initialize voice assistant
        
        Args:
            stt: Speech-to-text instance
            tts: Text-to-speech instance
            llm: Language model instance
        """
        self.stt = stt
        self.tts = tts
        self.llm = llm
        
        # Query queue for async handling
        self.query_queue = Queue()
        
        # Current vision context
        self.vision_context = {
            'recognized_people': [],
            'face_count': 0,
            'objects': []
        }
        
        # Assistant state
        self.is_active = False
        self.is_processing = False
        
        print("✅ Voice Assistant initialized")
    
    def update_vision_context(self, recognized_people=None, face_count=0, objects=None):
        """
        Update current vision context
        
        Args:
            recognized_people: List of recognized person names
            face_count: Number of faces detected
            objects: List of detected objects
        """
        if recognized_people is not None:
            self.vision_context['recognized_people'] = recognized_people
        
        self.vision_context['face_count'] = face_count
        
        if objects is not None:
            self.vision_context['objects'] = objects
    
    def handle_voice_command(self, transcript):
        """
        Handle voice command from STT
        
        Args:
            transcript: Transcribed text from user
        """
        if self.is_processing:
            print("⏳ Still processing previous command...")
            return
        
        self.is_processing = True
        
        try:
            print(f"\n🎤 User said: {transcript}")
            
            # Check for wake word or activation
            transcript_lower = transcript.lower()
            
            # Handle special commands
            if any(word in transcript_lower for word in ['exit', 'quit', 'goodbye', 'stop']):
                self.tts.speak("Goodbye!")
                self.is_active = False
                return
            
            if 'reset' in transcript_lower or 'clear' in transcript_lower:
                self.llm.reset_conversation()
                self.tts.speak("Conversation reset.")
                return
            
            # Get AI response
            response = self.llm.get_response(transcript, self.vision_context)
            print(f"🤖 AI: {response}")
            
            # Speak response
            self.tts.speak(response)
        
        except Exception as e:
            print(f"❌ Error handling command: {e}")
            self.tts.speak("Sorry, I encountered an error.")
        
        finally:
            self.is_processing = False
    
    async def start_listening(self):
        """Start listening for voice commands"""
        self.is_active = True
        
        # Set STT callback
        self.stt.callback = self.handle_voice_command
        
        # Start STT
        await self.stt.start_listening()
        
        print("🎤 Voice assistant active")
    
    async def stop_listening(self):
        """Stop listening"""
        self.is_active = False
        await self.stt.stop_listening()
        print("🛑 Voice assistant stopped")
    
    def speak(self, text):
        """Make assistant speak"""
        self.tts.speak(text)
    
    def greet_person(self, name):
        """Greet recognized person"""
        greetings = [
            f"Hello {name}! How can I help you today?",
            f"Hi {name}! Nice to see you!",
            f"Welcome back, {name}!",
            f"Good to see you, {name}!"
        ]
        
        import random
        greeting = random.choice(greetings)
        self.speak(greeting)


# Async helper for integration with sync code
class VoiceAssistantThread:
    """Run voice assistant in separate thread"""
    
    def __init__(self, voice_assistant):
        self.assistant = voice_assistant
        self.loop = None
        self.thread = None
    
    def start(self):
        """Start assistant in background thread"""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def _run_loop(self):
        """Run async loop in thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.assistant.start_listening())
        except Exception as e:
            print(f"❌ Voice assistant error: {e}")
    
    def stop(self):
        """Stop assistant"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.assistant.stop_listening(),
                self.loop
            )