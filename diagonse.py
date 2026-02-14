"""
Voice Assistant Diagnostic Script
Checks what's preventing voice assistant from loading
"""

import os
from dotenv import load_dotenv

print("="*70)
print("VOICE ASSISTANT DIAGNOSTIC")
print("="*70)

# Load environment
load_dotenv()
print("\n[1] Checking .env file...")
deepgram_key = os.getenv("DEEPGRAM_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if deepgram_key:
    print(f"  ✓ DEEPGRAM_API_KEY: {deepgram_key[:10]}...")
else:
    print("  ✗ DEEPGRAM_API_KEY not found!")

if gemini_key:
    print(f"  ✓ GEMINI_API_KEY: {gemini_key[:10]}...")
else:
    print("  ✗ GEMINI_API_KEY not found!")

# Check dependencies
print("\n[2] Checking dependencies...")

try:
    import deepgram
    print(f"  ✓ deepgram installed (version: {deepgram.__version__ if hasattr(deepgram, '__version__') else 'unknown'})")
except ImportError as e:
    print(f"  ✗ deepgram not installed: {e}")

try:
    import google.generativeai
    print("  ✓ google.generativeai installed")
except ImportError as e:
    print(f"  ✗ google.generativeai not installed: {e}")

try:
    import pygame
    print("  ✓ pygame installed")
except ImportError as e:
    print(f"  ✗ pygame not installed: {e}")

try:
    import requests
    print("  ✓ requests installed")
except ImportError as e:
    print(f"  ✗ requests not installed: {e}")

# Check file structure
print("\n[3] Checking file structure...")

files_to_check = [
    "vision/audio/__init__.py",
    "vision/audio/stt/__init__.py",
    "vision/audio/stt/deepgram_stt.py",
    "vision/audio/tts/__init__.py",
    "vision/audio/tts/deepgram_tts.py",
    "vision/audio/llm/__init__.py",
    "vision/audio/llm/gemini_llm.py",
    "vision/audio/voice_assistant.py"
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        if size > 0:
            print(f"  ✓ {filepath} ({size} bytes)")
        else:
            print(f"  ⚠ {filepath} (empty!)")
    else:
        print(f"  ✗ {filepath} (missing!)")

# Try to import modules
print("\n[4] Testing module imports...")

try:
    from vision.audio.stt.deepgram_stt import DeepgramSTT
    print("  ✓ DeepgramSTT imports successfully")
except Exception as e:
    print(f"  ✗ DeepgramSTT import failed: {e}")

try:
    from vision.audio.tts.deepgram_tts import DeepgramTTS
    print("  ✓ DeepgramTTS imports successfully")
except Exception as e:
    print(f"  ✗ DeepgramTTS import failed: {e}")

try:
    from vision.audio.llm.gemini_llm import GeminiLLM
    print("  ✓ GeminiLLM imports successfully")
except Exception as e:
    print(f"  ✗ GeminiLLM import failed: {e}")

try:
    from vision.audio.voice_assistant import VoiceAssistant
    print("  ✓ VoiceAssistant imports successfully")
except Exception as e:
    print(f"  ✗ VoiceAssistant import failed: {e}")

# Try to initialize components
print("\n[5] Testing component initialization...")

if deepgram_key and gemini_key:
    try:
        from vision.audio.tts.deepgram_tts import DeepgramTTS
        tts = DeepgramTTS(deepgram_key)
        print("  ✓ TTS initialized")
    except Exception as e:
        print(f"  ✗ TTS initialization failed: {e}")
    
    try:
        from vision.audio.llm.gemini_llm import GeminiLLM
        llm = GeminiLLM(gemini_key)
        print("  ✓ LLM initialized")
    except Exception as e:
        print(f"  ✗ LLM initialization failed: {e}")

def _init_voice_assistant(self):
    """Initialize voice assistant with detailed debugging"""
    print("\n  [DEBUG] Starting voice assistant initialization...")
    
    try:
        # Check API keys
        print("  [DEBUG] Checking API keys...")
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        print(f"  [DEBUG] Deepgram key: {'Found' if deepgram_key else 'NOT FOUND'}")
        print(f"  [DEBUG] Gemini key: {'Found' if gemini_key else 'NOT FOUND'}")
        
        if not deepgram_key or not gemini_key:
            print("  ✗ Voice disabled - API keys not set in .env")
            return
        
        # Import modules
        print("  [DEBUG] Importing STT module...")
        from vision.audio.stt.deepgram_stt import DeepgramSTT
        print("  [DEBUG] ✓ STT imported")
        
        print("  [DEBUG] Importing TTS module...")
        from vision.audio.tts.deepgram_tts import DeepgramTTS
        print("  [DEBUG] ✓ TTS imported")
        
        print("  [DEBUG] Importing LLM module...")
        from vision.audio.llm.gemini_llm import GeminiLLM
        print("  [DEBUG] ✓ LLM imported")
        
        print("  [DEBUG] Importing VoiceAssistant...")
        from vision.audio.voice_assistant import VoiceAssistant, VoiceAssistantThread
        print("  [DEBUG] ✓ VoiceAssistant imported")
        
        # Initialize components
        print("  [DEBUG] Creating STT instance...")
        stt = DeepgramSTT(deepgram_key)
        print("  [DEBUG] ✓ STT created")
        
        print("  [DEBUG] Creating TTS instance...")
        tts = DeepgramTTS(deepgram_key)
        print("  [DEBUG] ✓ TTS created")
        
        print("  [DEBUG] Creating LLM instance...")
        llm = GeminiLLM(gemini_key)
        print("  [DEBUG] ✓ LLM created")
        
        # Create voice assistant
        print("  [DEBUG] Creating VoiceAssistant...")
        assistant = VoiceAssistant(stt, tts, llm)
        print("  [DEBUG] ✓ Assistant created")
        
        print("  [DEBUG] Creating thread manager...")
        thread_manager = VoiceAssistantThread(assistant)
        print("  [DEBUG] ✓ Thread manager created")
        
        # Store in dict
        self.voice_assistant = {
            'assistant': assistant,
            'thread': thread_manager,
            'stt': stt,
            'tts': tts,
            'llm': llm
        }
        
        print("  ✓ Voice AI ready")
        print("    Press V to activate voice assistant")
        
    except ImportError as e:
        print(f"  ✗ Voice disabled - import error:")
        print(f"     {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"  ✗ Voice disabled - initialization error:")
        print(f"     {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nIf you see ✗ marks above, those are the issues to fix!")