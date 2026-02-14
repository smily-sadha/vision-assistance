import asyncio
import edge_tts
import tempfile
import os
import threading
import queue
import playsound
from audio.stt.stt_vad_engine import AudioState

class TTSEngine:

    def __init__(self, voice="en-US-AnaNeural"):

        self.voice = voice
        self.queue = queue.Queue()

        threading.Thread(target=self.worker, daemon=True).start()

    def speak(self, text):
        self.queue.put(text)

    def worker(self):

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            text = self.queue.get()
            loop.run_until_complete(self._speak_async(text))

    async def _speak_async(self, text):

        AudioState.speaking = True

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(path)

            playsound.playsound(path)

        finally:
            AudioState.speaking = False
            os.remove(path)
