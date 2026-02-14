import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import collections
from faster_whisper import WhisperModel
import time

class AudioState:
    speaking = False

class STTVADEngine:

    def __init__(self,
                 samplerate=16000,
                 frame_duration=30,
                 silence_timeout=1.0):

        self.samplerate = samplerate
        self.frame_duration = frame_duration
        self.frame_size = int(samplerate * frame_duration / 1000)

        self.vad = webrtcvad.Vad(2)  # Aggressiveness (0–3)

        self.audio_queue = queue.Queue()
        self.buffer = []
        self.last_voice_time = time.time()

        print("[STT] Loading Whisper model...")
        self.model = WhisperModel("base", compute_type="int8")
        print("✓ Whisper ready")

        self.silence_timeout = silence_timeout

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_size,
            callback=self.audio_callback
        )

    def audio_callback(self, indata, frames, time_info, status):

        if AudioState.speaking:
            return  # Ignore mic during TTS

        audio_frame = bytes(indata)

        is_speech = self.vad.is_speech(audio_frame, self.samplerate)

        current_time = time.time()

        if is_speech:
            self.buffer.append(np.frombuffer(audio_frame, dtype=np.int16))
            self.last_voice_time = current_time
        else:
            # Check silence duration
            if self.buffer and (current_time - self.last_voice_time > self.silence_timeout):
                self.audio_queue.put(np.concatenate(self.buffer))
                self.buffer.clear()

    def start(self):

        self.stream.start()
        threading.Thread(target=self.transcription_worker, daemon=True).start()

    def transcription_worker(self):

        while True:

            audio_data = self.audio_queue.get()

            try:
                audio_float = audio_data.astype(np.float32) / 32768.0

                segments, _ = self.model.transcribe(audio_float)

                text = " ".join([seg.text for seg in segments]).strip()

                if text:
                    print(f"[USER]: {text}")

                    self.on_transcription(text)

            except Exception as e:
                print("[STT ERROR]", e)

    def on_transcription(self, text):
        """Override this from pipeline / agent"""
        pass
