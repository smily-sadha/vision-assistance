"""
Google Gemini LLM Integration
Specialized for Vision-Impaired Assistance
"""

import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np


# Vision-impaired specialized prompt
VISION_IMPAIRED_PROMPT = """
You are an AI vision assistant specifically designed to help blind and visually impaired people.
Your role is to be their eyes and provide detailed, helpful descriptions of their surroundings.

CORE PRINCIPLES:
1. Be descriptive and specific - details matter
2. Use clear, natural language without jargon
3. Prioritize safety-relevant information
4. Be patient and encouraging
5. Speak conversationally, not robotically

DESCRIBING PEOPLE:
- Always mention if someone is present
- State their name if you recognize them
- Describe their location (e.g., "John is standing to your left, about 2 meters away")
- Mention any approaching people for safety

DESCRIBING OBJECTS:
- Focus on relevant objects that might affect navigation or tasks
- Mention location and distance when important
- Prioritize obstacles, hazards, or useful items
- Describe object positions relative to the user

SAFETY PRIORITIES:
- Always mention obstacles in the path
- Alert about open doors, stairs, or elevation changes
- Warn about potentially dangerous objects
- Note any approaching people or movement

INTERACTION STYLE:
- Keep responses 2-4 sentences unless more detail is requested
- Use natural, conversational tone
- Focus on actionable information
- Prioritize people first (social/safety priority)

IMPORTANT – VOICE OUTPUT RULES (always follow these):
- Your response will be spoken aloud by a text-to-speech engine.
- Keep every reply to 1-2 sentences maximum unless the user explicitly asks for more detail.
- Never use markdown, bullet points, numbered lists, asterisks, or special characters.
- Write as if you are speaking naturally in conversation.
- Avoid filler phrases like "Certainly!" or "Of course!" — get straight to the answer.

REMEMBER: You are providing critical information for independence and safety.
"""



class GeminiLLM:
    """Google Gemini AI for vision-impaired assistance"""
    
    def __init__(self, api_key, model_name="gemini-2.5-flash", 
                 vision_impaired_mode=True):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Google AI API key
            model_name: Model to use
            vision_impaired_mode: Use specialized vision-impaired prompts
        """
        self.api_key = api_key
        self.vision_impaired_mode = vision_impaired_mode
        
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        # Start conversation
        self.chat = self.model.start_chat(history=[])
        
        # Select system prompt
        if vision_impaired_mode:
            self.system_context = VISION_IMPAIRED_PROMPT
            print(f"✅ Gemini {model_name} (Vision-Impaired Mode)")
        else:
            self.system_context = "You are a helpful AI vision assistant."
            print(f"✅ Gemini {model_name} initialized")
    
    def get_response(self, user_query, context=None):
        """
        Get AI response to user query
        
        Args:
            user_query: User's question/command
            context: Current vision context
            
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
        """Build context-aware prompt for vision-impaired users"""
        prompt_parts = [self.system_context]
        
        if context and self.vision_impaired_mode:
            prompt_parts.append("\n=== CURRENT SCENE ===")
            
            # People (highest priority for vision-impaired)
            if 'recognized_people' in context and context['recognized_people']:
                people = ", ".join(context['recognized_people'])
                prompt_parts.append(f"People present: {people}")
                prompt_parts.append(f"Number of people: {len(context['recognized_people'])}")
            else:
                prompt_parts.append("People present: None detected")
            
            # Objects (for navigation/safety)
            if 'objects' in context and context['objects']:
                # Prioritize potentially hazardous or important objects
                important_objects = []
                all_objects = context['objects'][:10]
                
                # Safety-relevant objects first
                safety_keywords = ['chair', 'table', 'door', 'stairs', 'car', 
                                 'bicycle', 'bench', 'bottle']
                for obj in all_objects:
                    if any(keyword in obj.lower() for keyword in safety_keywords):
                        important_objects.append(obj)
                
                # Add remaining objects
                for obj in all_objects:
                    if obj not in important_objects:
                        important_objects.append(obj)
                
                objects_str = ", ".join(important_objects[:10])
                prompt_parts.append(f"Objects detected: {objects_str}")
            
            prompt_parts.append("===================\n")
        
        elif context:
            # Standard context (non vision-impaired mode)
            prompt_parts.append("\nCurrent Context:")
            
            if 'recognized_people' in context and context['recognized_people']:
                people = ", ".join(context['recognized_people'])
                prompt_parts.append(f"- People: {people}")
            
            if 'objects' in context and context['objects']:
                objects = ", ".join(context['objects'][:5])
                prompt_parts.append(f"- Objects: {objects}")
            
            if 'face_count' in context:
                prompt_parts.append(f"- Faces: {context['face_count']}")
        
        # User query
        prompt_parts.append(f"\nUser: {user_query}")
        
        # Response instruction
        if self.vision_impaired_mode:
            prompt_parts.append("\nRespond following vision-impaired assistance principles:")
        else:
            prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.chat = self.model.start_chat(history=[])
        print("🔄 Conversation reset")
    
    def set_vision_impaired_mode(self, enabled=True):
        """Toggle vision-impaired mode"""
        self.vision_impaired_mode = enabled
        mode = "Vision-Impaired" if enabled else "Standard"
        print(f"🔄 Mode: {mode}")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        # Initialize with vision-impaired mode
        llm = GeminiLLM(api_key, vision_impaired_mode=True)
        
        # Test with context
        context = {
            'recognized_people': ['Sarah', 'Mark'],
            'face_count': 2,
            'objects': ['chair', 'table', 'laptop', 'door', 'phone']
        }
        
        print("\n" + "="*70)
        print("VISION-IMPAIRED MODE TEST")
        print("="*70)
        
        queries = [
            "Who is here?",
            "What objects are around me?",
            "Is it safe to walk forward?",
            "Describe what you see"
        ]
        
        for query in queries:
            print(f"\n🎤 User: {query}")
            response = llm.get_response(query, context)
            print(f"🤖 AI: {response}")
            print("-"*70)
    else:
        print("Set GEMINI_API_KEY in .env file")