from easim.agents.prompt.base import BasePrompts


class ObjectNavPrompts(BasePrompts):
    """Prompt templates for Object Goal Navigation."""

    @property
    def system_prompt(self) -> str:
        return """You are an embodied AI agent navigating indoor environments to find specific objects.

Available actions:
- MOVE_FORWARD: Move forward by 0.25m
- TURN_LEFT: Turn left by 30 degrees  
- TURN_RIGHT: Turn right by 30 degrees
- LOOK_UP: Tilt camera up by 30 degrees
- LOOK_DOWN: Tilt camera down by 30 degrees
- STOP: End episode (call when target object found)

Always respond with exactly one action name. Think step by step about:
1. What objects do you see?
2. Where is your target object likely to be?
3. What's the best next action to find it?"""

    def generate_navigation_prompt(self, target_object: str, scene_description: str,
                                   detected_objects: list, action_history: list,
                                   position: list = None, heading: float = None) -> str:
        prompt = f"""
TARGET: Find a {target_object}

CURRENT SCENE: {scene_description}

DETECTED OBJECTS: {', '.join(detected_objects) if detected_objects else 'None visible'}

RECENT ACTIONS: {' -> '.join(action_history[-3:]) if action_history else 'None'}

POSITION: {position if position else 'Unknown'}
HEADING: {heading if heading else 'Unknown'}

What action should you take next to find the {target_object}?
Respond with exactly one action: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN, or STOP
"""
        return prompt