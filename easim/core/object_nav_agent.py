import random
from typing import List
from easim.core.types import Observation
from easim.core.habitat_wrapper import HabitatWrapper


class RandomObjectNavAgent:
    """Simple random agent for object navigation."""
    
    def __init__(self):
        """Initialize the random agent."""
        self.actions = ['move_forward', 'turn_left', 'turn_right']
        self.forward_prob = 0.7  # Bias towards moving forward
    
    def act(self, observation: Observation) -> str:
        """
        Choose an action based on the observation.
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Action name to take
        """
        # Simple random policy with forward bias
        if random.random() < self.forward_prob:
            return 'move_forward'
        else:
            return random.choice(['turn_left', 'turn_right'])
    
    def reset(self) -> None:
        """Reset agent state (nothing to reset for random agent)."""
        pass


class HeuristicObjectNavAgent:
    """Simple heuristic agent that tries to explore systematically."""
    
    def __init__(self):
        """Initialize the heuristic agent."""
        self.step_count = 0
        self.stuck_count = 0
        self.last_rgb = None
        self.exploration_steps = 0
        
    def act(self, observation: Observation) -> str:
        """
        Choose an action based on simple heuristics.
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Action name to take
        """
        self.step_count += 1
        
        # Check if we're stuck (similar observations)
        if self._is_stuck(observation):
            self.stuck_count += 1
            if self.stuck_count > 3:
                # Turn around when stuck
                self.stuck_count = 0
                return random.choice(['turn_left', 'turn_right'])
        else:
            self.stuck_count = 0
        
        # Simple exploration strategy
        if self.step_count % 20 == 0:
            # Periodically turn to explore
            return random.choice(['turn_left', 'turn_right'])
        
        # Default: move forward
        return 'move_forward'
    
    def _is_stuck(self, observation: Observation) -> bool:
        """Check if agent appears to be stuck."""
        if self.last_rgb is None:
            self.last_rgb = observation.rgb
            return False
        
        # Simple check: if RGB image is very similar, might be stuck
        diff = abs(observation.rgb.mean() - self.last_rgb.mean())
        self.last_rgb = observation.rgb
        
        return diff < 1.0  # Threshold for "similar" images
    
    def reset(self) -> None:
        """Reset agent state."""
        self.step_count = 0
        self.stuck_count = 0
        self.last_rgb = None
        self.exploration_steps = 0