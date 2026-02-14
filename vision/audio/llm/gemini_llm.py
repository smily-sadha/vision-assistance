"""
Google Gemini LLM Integration
Uses Gemini for intelligent responses based on vision context
"""

import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np


class GeminiLLM:
    """Google Gemini AI for intelligent responses"""
    
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Google AI API key
            model_name: Model to use (gemini-1.5-flash or gemini-1.5-pro)
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        # Conversation history
        self.chat = self.model.start_chat(history=[])
        
        # System context
        self.system_context = """
You are an AI vision assistant helping with face recognition and visual tasks.
You can see who is in the camera and provide helpful information.
Keep responses concise and natural (2-3 sentences max).
Be friendly and helpful.
"""
        
        print(f"✅ Gemini {model_name} initialized")
    
    def get_response(self, user_query, context=None):
        """
        Get AI response to user query
        
        Args:
            user_query: User's question/command
            context: Current vision context (recognized faces, objects, etc.)
            
        Returns:
            AI response text
        """
        try:
            # Build context-aware prompt
            prompt = self._build_prompt(user_query, context)
            
            # Get response
            response = self.chat.send_message(prompt)
            
            return response.text
        
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def get_response_with_image(self, user_query, image, context=None):
        """
        Get AI response with image input
        
        Args:
            user_query: User's question
            image: OpenCV image (BGR) or PIL Image
            context: Vision context
            
        Returns:
            AI response text
        """
        try:
            # Convert OpenCV to PIL if needed
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Build prompt
            prompt = self._build_prompt(user_query, context)
            
            # Get response with image
            response = self.model.generate_content([prompt, image])
            
            return response.text
        
        except Exception as e:
            print(f"❌ Gemini vision error: {e}")
            return "I couldn't analyze the image."
    
    def _build_prompt(self, user_query, context):
        """Build context-aware prompt"""
        prompt_parts = [self.system_context]
        
        if context:
            prompt_parts.append("\nCurrent Context:")
            
            if 'recognized_people' in context and context['recognized_people']:
                people = ", ".join(context['recognized_people'])
                prompt_parts.append(f"- Recognized people: {people}")
            
            if 'objects' in context and context['objects']:
                objects = ", ".join(context['objects'][:5])
                prompt_parts.append(f"- Detected objects: {objects}")
            
            if 'face_count' in context:
                prompt_parts.append(f"- Number of faces: {context['face_count']}")
        
        prompt_parts.append(f"\nUser: {user_query}")
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.chat = self.model.start_chat(history=[])
        print("🔄 Conversation reset")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Please set GEMINI_API_KEY in .env file")
        exit(1)
    
    llm = GeminiLLM(api_key)
    
    # Test without context
    print("\n=== Test 1: Simple query ===")
    response = llm.get_response("What can you help me with?")
    print(f"AI: {response}")
    
    # Test with context
    print("\n=== Test 2: With face recognition context ===")
    context = {
        'recognized_people': ['John', 'Sarah'],
        'face_count': 2
    }
    response = llm.get_response("Who is here?", context)
    print(f"AI: {response}")
    
    # Test with image (example)
    print("\n=== Test 3: With image ===")
    # Create a simple test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test Image", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    response = llm.get_response_with_image("What do you see in this image?", test_img)
    print(f"AI: {response}")