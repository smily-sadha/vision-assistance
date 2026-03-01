import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path("D:/D/final year project/vision_assistant/.env")

print("Exists:", env_path.exists())

load_dotenv(dotenv_path=env_path, override=True)

print("Deepgram raw:", os.getenv("DEEPGRAM_API_KEY"))
print("Gemini raw:", os.getenv("GEMINI_API_KEY"))