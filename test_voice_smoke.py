"""Quick smoke test: verify both classes import and instantiate correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from vision.audio.voice_assistant import VoiceAssistant, VoiceAssistantThread
from vision.audio.stt.deepgram_stt  import DeepgramSTT
from vision.audio.tts.deepgram_tts  import DeepgramTTS
from vision.audio.llm.gemini_llm    import GeminiLLM

dg_key  = os.getenv("DEEPGRAM_API_KEY")
gm_key  = os.getenv("GEMINI_API_KEY")

stt  = DeepgramSTT(dg_key)
tts  = DeepgramTTS(gm_key)
llm  = GeminiLLM(gm_key)

assistant   = VoiceAssistant(stt, tts, llm)
thread_mgr  = VoiceAssistantThread(assistant)

print("✅ Import + instantiation OK")
print(f"   VoiceAssistantThread methods: {[m for m in dir(thread_mgr) if not m.startswith('_')]}")
