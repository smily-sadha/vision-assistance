"""
Deepgram Text-to-Speech
Natural voice synthesis using Deepgram API
"""

import os
import tempfile
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

        # Fully initialize pygame before opening the mixer
        if not pygame.get_init():
            pygame.init()
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)

        print("✓ Deepgram TTS initialized")

    def speak(self, text, voice="aura-asteria-en"):
        """
        Convert text to speech and play it.

        Args:
            text: Text to speak
            voice: Deepgram Aura voice model
                   Options: aura-asteria-en (female, warm)
                            aura-luna-en   (female, bright)
                            aura-orion-en  (male, professional)
                            aura-zeus-en   (male, authoritative)
        """
        try:
            short_preview = text[:60] + ("..." if len(text) > 60 else "")
            print(f"🔊 Speaking: {short_preview}")

            url = "https://api.deepgram.com/v1/speak"
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            }
            params = {
                "model": voice,
                "encoding": "linear16",
                "container": "wav",
            }
            data = {"text": text}

            response = requests.post(
                url, headers=headers, params=params, json=data, timeout=15
            )

            if response.status_code == 200:
                # Write to a temp WAV file.
                # pygame on Windows cannot reliably load audio from a BytesIO
                # object, so we write the response to disk first.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                try:
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()

                    # Block until playback finishes
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)

                    print("✓ Finished speaking")
                finally:
                    pygame.mixer.music.unload()
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            else:
                print(f"❌ TTS API error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            print(f"❌ TTS error: {e}")

    def save_to_file(self, text, filename, voice="aura-asteria-en"):
        """
        Save TTS speech to a WAV file on disk.

        Args:
            text: Text to convert
            filename: Output filename (.wav)
            voice: Voice model
        """
        try:
            url = "https://api.deepgram.com/v1/speak"
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            }
            params = {
                "model": voice,
                "encoding": "linear16",
                "container": "wav",
            }
            data = {"text": text}

            response = requests.post(
                url, headers=headers, params=params, json=data, timeout=15
            )

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✓ Saved: {filename}")
            else:
                print(f"❌ Save error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            print(f"❌ Save error: {e}")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("DEEPGRAM_API_KEY")

    if api_key:
        tts = DeepgramTTS(api_key)
        tts.speak("Hello! I am your AI vision assistant.")
    else:
        print("Set DEEPGRAM_API_KEY in .env file")