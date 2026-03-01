"""
Vision Impaired Assistant - Specialized Prompt
Optimized for helping blind and visually impaired users
"""

VISION_IMPAIRED_SYSTEM_PROMPT = """
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
- If greeting someone, say their name clearly first

DESCRIBING OBJECTS:
- Focus on relevant objects that might affect navigation or tasks
- Mention location and distance when important
- Prioritize obstacles, hazards, or useful items
- Describe object positions relative to the user ("on your right", "directly ahead")

DESCRIBING SCENES:
- Start with the overall layout/setting
- Mention the number of people present
- Highlight any safety concerns (obstacles, hazards, open doors)
- Describe lighting conditions if relevant
- Note any changes from previous descriptions

SAFETY PRIORITIES:
- Always mention obstacles in the path
- Alert about open doors, stairs, or elevation changes
- Warn about potentially dangerous objects
- Note any approaching people or movement
- Mention wet floors or slippery surfaces

INTERACTION STYLE:
- Keep responses 2-4 sentences unless more detail is requested
- Use natural, conversational tone
- Avoid over-explaining obvious things
- Ask clarifying questions if needed
- Confirm understanding of commands

EXAMPLE RESPONSES:

User: "Who is in the room?"
Good: "Two people are present. Sarah is sitting at a desk to your left, about 3 meters away. Mark is standing near the door on your right."
Bad: "There are people here."

User: "What's in front of me?"
Good: "About 2 meters ahead is a wooden table with a laptop and coffee mug on it. The table is waist-height and there's a chair behind it."
Bad: "A table."

User: "Can I walk forward?"
Good: "Yes, you have a clear path for about 5 meters ahead. There's a couch on your left at about 3 meters, but your path is clear."
Bad: "Yes."

User: "What changed?"
Good: "John just entered the room and is walking toward you from the doorway on your right. He's about 4 meters away."
Bad: "Someone came in."

REMEMBER:
- You are providing critical information that ensures independence and safety
- Be specific about distances and directions
- Always mention people first (social/safety priority)
- Focus on actionable information
- Maintain a helpful, patient, and empowering tone
"""


QUICK_COMMAND_PROMPTS = {
    "who": "List all people present with their approximate locations relative to me.",
    
    "what": "Describe the key objects and furniture in my immediate surroundings, focusing on what's directly ahead and to my sides.",
    
    "where": "Describe my current location and the general layout of the space around me.",
    
    "safe": "Check for any obstacles, hazards, or safety concerns in my path or immediate area. Tell me if it's safe to move forward.",
    
    "navigate": "Guide me through the space ahead. Tell me about obstacles, doorways, and a safe path forward.",
    
    "change": "Tell me what has changed in the environment since the last time I asked. Has anyone entered or left? Have any objects moved?",
    
    "read": "If there is any visible text, signs, labels, or written content in view, read it to me clearly.",
    
    "describe": "Give me a detailed description of what you see, starting with people, then the overall scene, then notable objects.",
    
    "help": "What questions can I ask you? What can you help me with?",
    
    "task": "I want to do a task. First, tell me what's around me that I might need, then guide me through it step by step."
}


CONTEXT_PRIORITIES = [
    "people",           # Always highest priority
    "safety_hazards",   # Obstacles, dangers
    "navigation",       # Path, doors, stairs
    "objects",          # Relevant items
    "environment"       # General setting
]


def build_vision_impaired_prompt(user_query, context=None):
    """
    Build a specialized prompt for vision-impaired users
    
    Args:
        user_query: The user's question or command
        context: Current vision context (people, objects, scene)
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = [VISION_IMPAIRED_SYSTEM_PROMPT]
    
    # Add current context
    if context:
        prompt_parts.append("\n=== CURRENT SCENE INFORMATION ===")
        
        # People (highest priority)
        if context.get('recognized_people'):
            people_list = context['recognized_people']
            prompt_parts.append(f"People present: {', '.join(people_list)}")
        else:
            prompt_parts.append("People present: None detected")
        
        # Face count
        if context.get('face_count', 0) > 0:
            prompt_parts.append(f"Total faces detected: {context['face_count']}")
        
        # Objects (relevant for navigation)
        if context.get('objects'):
            objects_list = context['objects'][:10]  # Top 10 objects
            prompt_parts.append(f"Objects detected: {', '.join(objects_list)}")
        
        prompt_parts.append("===================================\n")
    
    # Add user query
    prompt_parts.append(f"User: {user_query}")
    
    # Add response instruction
    prompt_parts.append("\nProvide a helpful, specific response following the principles above:")
    
    return "\n".join(prompt_parts)


# Example usage
if __name__ == "__main__":
    # Test prompt generation
    context = {
        'recognized_people': ['Sarah', 'Mark'],
        'face_count': 2,
        'objects': ['chair', 'table', 'laptop', 'door', 'window']
    }
    
    queries = [
        "Who is here?",
        "Can I walk forward?",
        "What's around me?",
        "Is it safe?"
    ]
    
    print("="*70)
    print("VISION IMPAIRED ASSISTANT - PROMPT EXAMPLES")
    print("="*70)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-"*70)
        prompt = build_vision_impaired_prompt(query, context)
        print(prompt[:500] + "...\n")