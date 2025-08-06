import openai
import numpy as np
from typing import Dict, Any
from easim.agents.base import BaseLLMAgent
from easim.agents.prompt.eqa import EQAPrompts
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class EQAAgent(BaseLLMAgent):
    """LLM-based EQA agent using actual observations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prompts = EQAPrompts()

    def process_observations(self, obs) -> Dict[str, Any]:
        """Process raw observations from Habitat environment."""
        processed = {}
        
        # Get question tokens (always available in EQA)
        processed['question_tokens'] = obs.get('question', [])
        
        # Get spatial information
        processed['heading'] = float(obs['compass'][0]) if 'compass' in obs else None
        processed['position'] = obs['gps'].tolist() if 'gps' in obs else None
            
        return processed

    def generate_prompt(self, processed_obs: Dict[str, Any]) -> str:
        """Generate task-specific prompt."""
        return self.prompts.generate_eqa_prompt(
            question_tokens=processed_obs.get('question_tokens', []),
            scene_description=processed_obs.get('visual_description', ''),
            detected_objects=processed_obs.get('detected_objects', []),
            scene_analysis=processed_obs.get('scene_analysis', ''),
            action_history=self.action_history[-5:],  # Last 5 actions
            position=processed_obs.get('position'),
            heading=processed_obs.get('heading')
        )
    
    def parse_action(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into Habitat action."""
        response = llm_response.strip().upper()
        
        # Check if agent wants to answer
        if "ANSWER:" in response or "ANSWER " in response or response.startswith("ANSWER"):
            # Extract answer text (simplified - assumes single word answers)
            answer_text = response.split("ANSWER")[-1].strip(": ")
            
            # Map common answers to answer IDs (simplified)
            answer_map = {
                "RED": 0, "BLUE": 1, "GREEN": 2, "YELLOW": 3, "WHITE": 4, "BLACK": 5,
                "BROWN": 6, "PURPLE": 7, "ORANGE": 8, "PINK": 9
            }
            
            answer_id = answer_map.get(answer_text.split()[0], 0)
            return {
                "action": "answer",
                "action_args": {"answer_id": answer_id}
            }
        
        # Navigation actions
        action_map = {
            'MOVE_FORWARD': HabitatSimActions.move_forward,
            'TURN_LEFT': HabitatSimActions.turn_left, 
            'TURN_RIGHT': HabitatSimActions.turn_right,
            'LOOK_UP': HabitatSimActions.look_up,
            'LOOK_DOWN': HabitatSimActions.look_down
        }
        
        # Find matching action in response
        for action_name, action_value in action_map.items():
            if action_name in response:
                return {"action": action_value}
                
        # Default action if parsing fails
        return {"action": HabitatSimActions.move_forward}