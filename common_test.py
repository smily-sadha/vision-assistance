from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path('.') / '.env', override=True)

print("DEBUG DEEPGRAM:", repr(os.getenv("DEEPGRAM_API_KEY")))
print("DEBUG GEMINI:", repr(os.getenv("GEMINI_API_KEY")))
