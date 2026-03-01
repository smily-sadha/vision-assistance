import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

try:
    with open('models.txt', 'w', encoding='utf-8') as f:
        f.write("Available Models:\n")
        for m in genai.list_models():
             if 'generateContent' in m.supported_generation_methods:
                 f.write(m.name + '\n')
except Exception as e:
    with open('models.txt', 'w', encoding='utf-8') as f:
        f.write("Error: " + str(e))
