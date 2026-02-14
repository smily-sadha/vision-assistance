"""
Deepgram Text-to-Speech (Compatible with SDK 2.x and 3.x)
"""

import os
import io
import pygame
import requests


class DeepgramTTS:
    """Deepgram text-to-speech"""
    
    def __init__(self, api_key):
        """
        Initialize Deepgram TTS
        
        Args:
            api_key: Deepgram API key
        """
        self.api_key = api_key
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        print("✓ Deepgram TTS initialized")
    
    def speak(self, text, voice="aura-asteria-en"):
        """
        Convert text to speech and play (using REST API directly)
        
        Args:
            text: Text to speak
            voice: Voice model to use
        """
        try:
            print(f"🔊 Speaking: {text[:50]}...")
            
            # Use REST API directly (works with any SDK version)
            url = "https://api.deepgram.com/v1/speak"
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "model": voice,
                "encoding": "linear16",
                "container": "wav"
            }
            
            data = {
                "text": text
            }
            
            # Make request
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=data,
                stream=True
            )
            
            if response.status_code == 200:
                # Get audio data
                audio_data = b""
                for chunk in response.iter_content(chunk_size=1024):
                    audio_data += chunk
                
                # Play audio
                self._play_audio(audio_data)
                print("✅ Finished speaking")
            else:
                print(f"❌ TTS error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    def _play_audio(self, audio_data):
        """Play audio using pygame"""
        try:
            # Load audio from bytes
            audio_file = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_file)
            
            # Play
            pygame.mixer.music.play()
            
            # Wait until finished
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    def save_to_file(self, text, filename, voice="aura-asteria-en"):
        """
        Save speech to audio file
        
        Args:
            text: Text to convert
            filename: Output filename (.wav)
            voice: Voice model
        """
        try:
            url = "https://api.deepgram.com/v1/speak"
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "model": voice,
                "encoding": "linear16",
                "container": "wav"
            }
            
            data = {"text": text}
            
            response = requests.post(url, headers=headers, params=params, json=data, stream=True)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"✅ Saved to: {filename}")
            else:
                print(f"❌ Save error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Save error: {e}")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not api_key:
        print("❌ Please set DEEPGRAM_API_KEY in .env file")
        exit(1)
    
    tts = DeepgramTTS(api_key)
    
    # Test speech
    tts.speak("Hello! I am your AI vision assistant. How can I help you today?")