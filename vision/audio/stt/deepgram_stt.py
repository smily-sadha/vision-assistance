"""
Deepgram Speech-to-Text
Real-time speech recognition using Deepgram API
"""

import asyncio
import json


class DeepgramSTT:
    """Deepgram real-time speech-to-text"""
    
    def __init__(self, api_key, callback=None):
        """
        Initialize Deepgram STT
        
        Args:
            api_key: Deepgram API key
            callback: Function to call with transcribed text
        """
        self.api_key = api_key
        self.callback = callback
        self.is_listening = False
        
        print("✓ Deepgram STT initialized")
    
    async def start_listening(self):
        """Start listening to microphone"""
        self.is_listening = True
        print("🎤 Listening...")
    
    async def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        print("🛑 Stopped listening")
    
    def process_audio(self, audio_data):
        """Process audio and return transcription"""
        # For now, this is a placeholder
        # In full implementation, this would send to Deepgram API
        pass


# Example usage
if __name__ == "__main__":
    def test_callback(text):
        print(f"Transcribed: {text}")
    
    stt = DeepgramSTT("test_key", callback=test_callback)
    print("STT module loaded successfully!")