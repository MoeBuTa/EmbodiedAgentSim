import openai
from typing import Dict, Any
from easim.agents.base import BaseLLMAgent
from easim.agents.prompt.objectnav import ObjectNavPrompts


class ObjectNavAgent(BaseLLMAgent):
    """LLM agent using OpenAI's API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prompts = ObjectNavPrompts()

    def generate_prompt(self, processed_obs: Dict[str, Any]) -> str:
        """Generate task-specific prompt."""
        return self.prompts.generate_navigation_prompt(
            target_object=processed_obs.get('target_object', 'unknown'),
            scene_description=processed_obs.get('visual_description', ''),
            detected_objects=processed_obs.get('detected_objects', []),
            action_history=self.action_history[-5:],  # Last 5 actions
            position=processed_obs.get('position'),
            heading=processed_obs.get('heading')
        )
