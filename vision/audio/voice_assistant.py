"""
Voice Assistant Coordinator
Manages STT, TTS, and LLM integration with a continuous interactive loop.
"""

import threading
import time
from queue import Queue
import random


# ---------------------------------------------------------------------------
# Exit phrases that terminate the voice loop
# ---------------------------------------------------------------------------
EXIT_PHRASES = {
    "exit voice mode", "stop listening", "stop voice",
    "goodbye", "bye", "quit", "exit",
}

# Minimum Deepgram confidence score to accept a transcript
MIN_CONFIDENCE = 0.60


class VoiceAssistant:
    """AI Voice Assistant with vision capabilities"""

    def __init__(self, stt, tts, llm):
        self.stt = stt
        self.tts = tts
        self.llm = llm

        self.vision_context = {
            'recognized_people': [],
            'face_count': 0,
            'objects': []
        }

        self.is_active = False
        self.is_processing = False

        print("✅ Voice Assistant initialized")

    def update_vision_context(self, recognized_people=None, face_count=0, objects=None):
        """Update the current vision context sent to the LLM."""
        if recognized_people is not None:
            self.vision_context['recognized_people'] = recognized_people
        self.vision_context['face_count'] = face_count
        if objects is not None:
            self.vision_context['objects'] = objects

    def handle_voice_command(self, transcript):
        """Process a transcribed utterance: call LLM and speak the response."""
        if self.is_processing:
            return

        self.is_processing = True
        try:
            print(f"\n🎤 You: {transcript}")
            response = self.llm.get_response(transcript, self.vision_context)
            print(f"🤖 AI : {response}")
            self.tts.speak(response)
        except Exception as e:
            print(f"❌ Command error: {e}")
            self.tts.speak("Sorry, something went wrong.")
        finally:
            self.is_processing = False

    def speak(self, text):
        """Speak a piece of text directly."""
        try:
            self.tts.speak(text)
        except Exception as e:
            print(f"❌ Speak error: {e}")

    def greet_person(self, name):
        """Greet a recognised person."""
        greetings = [
            f"Hello {name}! I am ONI, your assistant. How can I help you today?",
            f"Hi {name}! ONI here. Nice to see you!",
            f"Welcome back, {name}! I am ONI, ready to assist.",
            f"Good to see you, {name}! ONI is online."
        ]
        self.speak(random.choice(greetings))

    def process_command(self, command):
        """Process a text command (non-voice path)."""
        try:
            print(f"\n💬 Command: {command}")
            response = self.llm.get_response(command, self.vision_context)
            print(f"🤖 AI: {response}")
            self.speak(response)
        except Exception as e:
            print(f"❌ Command error: {e}")


# ---------------------------------------------------------------------------
# Continuous voice thread
# ---------------------------------------------------------------------------

class VoiceAssistantThread:
    """
    Runs a continuous *listen → STT → LLM → TTS → listen* loop in a
    background thread.

    Behaviour:
    - Silence detection: 1.5 s of quiet after speech → send utterance to STT
    - 10 s of no speech detected → speak "Are you still there?"
    - Low Deepgram confidence (<0.60) → ask user to repeat
    - Exit phrases ("exit voice mode", "stop listening", …) → stop loop
    - After TTS finishes, immediately returns to listening (no key needed)
    - No audio is captured while TTS is playing (prevents echo)

    Public API:
        thread_mgr.start(on_stopped_callback=None)  – start background thread
        thread_mgr.stop()                            – force-stop the thread
    """

    # ── Audio settings ──────────────────────────────────────────────────────
    CHUNK    = 4096   # samples per pyaudio read
    CHANNELS = 1      # mono
    RATE     = 16000  # Hz

    # ── VAD / timing settings ───────────────────────────────────────────────
    SILENCE_THRESHOLD  = 400    # RMS below this = silence
    SILENCE_SECONDS    = 1.5    # seconds of silence to commit an utterance
    IDLE_TIMEOUT       = 10.0   # seconds of total silence → "Are you still there?"
    MIN_SPEECH_SECONDS = 0.4    # discard bursts shorter than this

    def __init__(self, assistant: VoiceAssistant):
        self.assistant        = assistant
        self._thread          = None
        self._stop_event      = threading.Event()
        self._on_stopped      = None   # optional callback when loop exits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, on_stopped_callback=None, on_learn_face_callback=None):
        """
        Start the continuous voice loop in a daemon thread.

        Args:
            on_stopped_callback: zero-arg callable invoked when the loop ends
                                 (e.g. on exit command). Use this to set
                                 voice_active = False in the main app.
            on_learn_face_callback: callable(name) invoked when user asks to learn a face
        """
        if self._thread and self._thread.is_alive():
            print("⚠ Voice thread already running")
            return

        self._on_stopped = on_stopped_callback
        self._on_learn_face = on_learn_face_callback
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="VoiceContinuousLoop"
        )
        self._thread.start()
        print("🎤 Voice assistant thread started")

    def stop(self):
        """Force-stop the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=4)
            self._thread = None
        print("🛑 Voice assistant thread stopped")

    # ------------------------------------------------------------------
    # Internal – thread entry point
    # ------------------------------------------------------------------

    def _run(self):
        try:
            import pyaudio
        except ImportError:
            print("❌ pyaudio not installed – run: pip install pyaudio")
            return

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            pa.terminate()
            return

        print("🎙️  Listening…  (say 'exit voice mode' to stop)")
        try:
            self._continuous_loop(stream)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            if self._on_stopped:
                self._on_stopped()

    # ------------------------------------------------------------------
    # Core continuous loop (state machine)
    # ------------------------------------------------------------------

    def _continuous_loop(self, stream):
        import numpy as np

        print("🎙️  Calibrating environment noise... please stay silent for 1s.")
        calibration_samples = []
        for _ in range(int(self.RATE / self.CHUNK)): # 1 second of audio
            try:
                raw = stream.read(self.CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.int16)
                rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                calibration_samples.append(rms)
            except Exception:
                pass
                
        if calibration_samples:
            bg_noise = sum(calibration_samples) / len(calibration_samples)
        else:
            bg_noise = 0
            
        # Set dynamic threshold at least 60, or average noise + 50
        dynamic_threshold = max(60, int(bg_noise + 50))
        print(f"🎙️  Noise level: {bg_noise:.1f}. Setting VAD threshold to: {dynamic_threshold}")

        silence_chunks_needed = int(
            self.SILENCE_SECONDS * self.RATE / self.CHUNK
        )
        min_speech_chunks = int(
            self.MIN_SPEECH_SECONDS * self.RATE / self.CHUNK
        )
        idle_chunks_needed = int(
            self.IDLE_TIMEOUT * self.RATE / self.CHUNK
        )

        audio_buffer   = []
        silence_count  = 0
        idle_count     = 0
        speaking       = False
        idle_prompted  = False   # prevents repeated "Are you still there?"

        while not self._stop_event.is_set():
            # ── read one chunk ────────────────────────────────────────
            try:
                raw = stream.read(self.CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            samples = np.frombuffer(raw, dtype=np.int16)
            rms     = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

            # ── speech detected ───────────────────────────────────────
            if rms > dynamic_threshold:
                speaking      = True
                silence_count = 0
                idle_count    = 0
                idle_prompted = False
                audio_buffer.append(raw)

            # ── silence while we were speaking ────────────────────────
            elif speaking:
                silence_count += 1
                audio_buffer.append(raw)

                if silence_count >= silence_chunks_needed:
                    # Utterance complete – process it
                    if len(audio_buffer) >= min_speech_chunks:
                        self._process_utterance(audio_buffer)
                        if self._stop_event.is_set():
                            break

                    audio_buffer  = []
                    silence_count = 0
                    speaking      = False

            # ── idle (no speech at all) ───────────────────────────────
            else:
                idle_count += 1
                if idle_count >= idle_chunks_needed and not idle_prompted:
                    idle_prompted = True
                    idle_count    = 0
                    print("💬 [idle prompt] Are you still there?")
                    self.assistant.speak("Are you still there?")

    def _process_utterance(self, frames):
        """Run STT → check exit / confidence → LLM → TTS."""
        import re
        result     = self._send_to_deepgram(frames)
        transcript = result.get("transcript", "").strip()
        confidence = result.get("confidence", 1.0)

        if not transcript:
            return   # empty – stay silent and keep listening

        # ── exit command? ─────────────────────────────────────────────
        if any(phrase in transcript.lower() for phrase in EXIT_PHRASES):
            print(f"🛑 Exit command: '{transcript}'")
            self.assistant.speak("Sure, turning off voice mode. Goodbye!")
            self._stop_event.set()
            return
            
        # ── learn face command? ───────────────────────────────────────
        learn_match = re.search(r"(?:this is|learn face) (?:my friend |the person )?([a-zA-Z0-9_]+)", transcript, re.IGNORECASE)
        if learn_match:
            name = learn_match.group(1).lower().strip()
            print(f"📸 Learn Face command: '{transcript}' -> Name: {name}")
            self.assistant.speak(f"Okay, I will learn the face for {name}. Please look at the camera for the next few seconds.")
            if getattr(self, "_on_learn_face", None):
                self._stop_event.set() # Stop the voice loop so camera can be used safely
                self._on_learn_face(name)
            return

        # ── low confidence? ───────────────────────────────────────────
        if confidence < MIN_CONFIDENCE:
            print(f"⚠ Low confidence ({confidence:.2f}): '{transcript}'")
            self.assistant.speak("Sorry, I didn't catch that. Could you repeat?")
            return

        # ── normal command ────────────────────────────────────────────
        self.assistant.handle_voice_command(transcript)

    # ------------------------------------------------------------------
    # Deepgram pre-recorded STT (REST)
    # ------------------------------------------------------------------

    def _send_to_deepgram(self, frames):
        """
        Encode PCM frames as WAV and POST to Deepgram /v1/listen.

        Returns:
            dict with keys 'transcript' (str) and 'confidence' (float).
        """
        import io
        import wave
        import requests

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))
        buf.seek(0)

        try:
            url     = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {self.assistant.stt.api_key}",
                "Content-Type": "audio/wav",
            }
            params  = {
                "model":        "nova-2",
                "smart_format": "true",
                "punctuate":    "true",
                "confidence":   "true",
            }
            resp = requests.post(
                url, headers=headers, params=params,
                data=buf.read(), timeout=10,
            )

            if resp.status_code == 200:
                data         = resp.json()
                alternative  = (
                    data.get("results", {})
                        .get("channels", [{}])[0]
                        .get("alternatives", [{}])[0]
                )
                transcript = alternative.get("transcript", "").strip()
                confidence = float(alternative.get("confidence", 1.0))
                if transcript:
                    print(f"📝 STT [{confidence:.2f}]: {transcript}")
                return {"transcript": transcript, "confidence": confidence}
            else:
                print(f"❌ Deepgram STT error {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            print(f"❌ STT request failed: {e}")

        return {"transcript": "", "confidence": 0.0}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Voice Assistant module loaded successfully!")
    print("Classes: VoiceAssistant, VoiceAssistantThread")