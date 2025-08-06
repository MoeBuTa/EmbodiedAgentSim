from easim.agents.prompt.base import BasePrompts


class EQAPrompts(BasePrompts):
    """Prompt templates for Embodied Question Answering."""

    @property
    def system_prompt(self) -> str:
        return """You are an embodied AI agent navigating indoor environments to answer questions.

Available actions:
- MOVE_FORWARD: Move forward by 0.25m
- TURN_LEFT: Turn left by 30 degrees  
- TURN_RIGHT: Turn right by 30 degrees
- LOOK_UP: Tilt camera up by 30 degrees
- LOOK_DOWN: Tilt camera down by 30 degrees
- ANSWER: [color/object] - Answer the question with a specific response

Your goal is to navigate the environment to find the information needed to answer the question, then provide the answer.

Always respond with exactly one action. Think step by step about:
1. What is the question asking?
2. What do you see in the current scene?
3. Do you have enough information to answer, or do you need to explore more?
4. What's the best next action?"""

    def generate_eqa_prompt(self, question_tokens: list, scene_description: str,
                           detected_objects: list, scene_analysis: str,
                           action_history: list, position: list = None, 
                           heading: float = None) -> str:
        
        # Convert question tokens to readable text (simplified - would need proper tokenizer)
        question_text = f"Question tokens: {question_tokens}" if question_tokens else "Visual question about the scene"
        
        prompt = f"""
QUESTION: {question_text}

RECENT ACTIONS: {' -> '.join(action_history[-3:]) if action_history else 'None'}

POSITION: {position if position else 'Unknown'}
HEADING: {heading if heading else 'Unknown'}

Look at the current scene in the image. Based on what you can see, can you answer the question?
If you can see enough to answer, respond with: ANSWER: [your answer]
If you need to explore more or get a better view, choose: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, or LOOK_DOWN

For color questions, look carefully at the objects in the scene and their colors.
For location questions, observe the room type and object placement.
"""
        return prompt