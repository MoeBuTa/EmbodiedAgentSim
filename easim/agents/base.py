from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import openai

from easim.agents.prompt.base import BasePrompts
from habitat.core.agent import Agent
from habitat.core.simulator import Observations


class BaseLLMAgent(Agent, ABC):
    """Base class for LLM-based embodied agents in Habitat."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.action_history = []
        self.observation_history = []
        self.spatial_memory = {}
        openai.api_key = config.get('openai_api_key')
        self.model = config.get('model', 'gpt-4o-mini')
        self.prompts = BasePrompts()


    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.action_history = []
        self.observation_history = []
        self.spatial_memory = {}

    def act(self, observations: Observations) -> Dict[str, Any]:
        """Main action selection method."""
        # Process observations
        processed_obs = self.process_observations(observations)

        # Generate LLM prompt
        prompt = self.generate_prompt(processed_obs)

        # Get LLM response
        llm_response = self.query_llm(prompt)

        # Parse response to action
        action = self.parse_action(llm_response)

        # Update memory
        self.action_history.append(action)
        self.observation_history.append(processed_obs)
        self.spatial_memory.update(processed_obs.get('spatial_memory', {}))

        return action

    def query_llm(self, prompt: str) -> str:
        """Query the LLM with given prompt."""
        """Query OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompts.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM query failed: {e}")
            return "MOVE_FORWARD"  # Default action

    def process_observations(self, obs: Observations) -> Dict[str, Any]:
        """Process raw observations into LLM-friendly format."""
        pass

    def generate_prompt(self, processed_obs: Dict[str, Any]) -> str:
        """Generate prompt for LLM based on observations."""
        # Implementation depends on your prompting strategy
        pass

    def parse_action(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into Habitat action."""
        # Map LLM output to Habitat actions
        pass