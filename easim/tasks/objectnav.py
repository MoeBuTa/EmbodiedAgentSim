"""Object Navigation task implementation for habitat-lab framework"""
from typing import Dict, Any, Optional, List
import numpy as np

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

from habitat.sims.habitat_simulator.actions import HabitatSimActions, HabitatSimActionsSingleton

from .base import BaseEmbodiedTask


class ObjectNavTask(BaseEmbodiedTask):
    """Object Navigation task using habitat-lab framework"""

    def __init__(self, dataset: str = "HM3D", **kwargs):
        """
        Initialize ObjectNav task
        
        Args:
            dataset: Dataset to use ("HM3D" or "MP3D")
            **kwargs: Additional arguments passed to base class
        """
        # Choose config based on dataset
        if dataset.upper() == "HM3D":
            config_path = "benchmark/nav/objectnav/objectnav_hm3d.yaml"
        elif dataset.upper() == "MP3D":
            config_path = "benchmark/nav/objectnav/objectnav_mp3d.yaml"
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        self.dataset = dataset
        self.target_object = None
        self.initial_distance_to_goal = None
        
        # ObjectNav specific parameters
        kwargs.setdefault('max_steps', 500)
        kwargs.setdefault('success_distance', 0.1)
        
        super().__init__(config_path, **kwargs)

    def _on_reset(self):
        """ObjectNav-specific reset logic"""
        # Get target object information if available
        if hasattr(self.env.current_episode, 'object_category'):
            self.target_object = self.env.current_episode.object_category
        elif 'objectgoal' in self.current_observations:
            self.target_object = self.current_observations['objectgoal']
        else:
            self.target_object = "unknown"
        
        # Get initial distance to goal
        if 'distance_to_goal' in self.current_observations:
            self.initial_distance_to_goal = self.current_observations['distance_to_goal']
        else:
            self.initial_distance_to_goal = float('inf')
        
        print(f"New episode started - Target object: {self.target_object}")
        if self.initial_distance_to_goal != float('inf'):
            print(f"Initial distance to goal: {self.initial_distance_to_goal:.2f}m")

    def _process_action(self, action: Any) -> Any:
        """Process and validate action for ObjectNav"""
        # If action is already a HabitatSimActions enum, return as is
        if isinstance(action, HabitatSimActionsSingleton):
            return action
        
        # If action is a string, convert to enum
        if isinstance(action, str):
            action_map = {
                'move_forward': HabitatSimActions.move_forward,
                'turn_left': HabitatSimActions.turn_left,
                'turn_right': HabitatSimActions.turn_right,
                'stop': HabitatSimActions.stop
            }
            if action in action_map:
                return action_map[action]
            else:
                raise ValueError(f"Unknown action: {action}")
        
        # If action is an integer, convert to enum
        if isinstance(action, int):
            actions = [
                HabitatSimActions.stop,
                HabitatSimActions.move_forward,
                HabitatSimActions.turn_left,
                HabitatSimActions.turn_right
            ]
            if 0 <= action < len(actions):
                return actions[action]
            else:
                raise ValueError(f"Action index out of range: {action}")
        
        # Default: return as is and let habitat-lab handle it
        return action

    def _is_episode_done(self) -> bool:
        """Check if ObjectNav episode should end"""
        # Episode is done if habitat-lab says so or max steps reached
        return self.is_episode_over or self.step_count >= self.max_steps

    def _update_episode_stats(self):
        """Update ObjectNav-specific episode statistics"""
        # Update basic stats
        self.episode_stats['steps'] = self.step_count
        
        # Update distance to goal if available
        if 'distance_to_goal' in self.current_observations:
            self.episode_stats['distance_to_goal'] = float(self.current_observations['distance_to_goal'])
        
        # Check for success
        self.episode_stats['success'] = self.evaluate_success()
        
        # Calculate path efficiency if we have initial distance
        if (self.initial_distance_to_goal != float('inf') and 
            self.initial_distance_to_goal > 0):
            current_distance = self.episode_stats.get('distance_to_goal', float('inf'))
            if current_distance != float('inf'):
                self.episode_stats['path_efficiency'] = self.initial_distance_to_goal / max(current_distance, 0.01)

    def _get_step_info(self) -> Dict[str, Any]:
        """Get ObjectNav-specific step information"""
        info = {
            'step': self.step_count,
            'target_object': self.target_object,
            'episode_over': self.is_episode_over,
        }
        
        # Add goal information if available
        if 'distance_to_goal' in self.current_observations:
            info['distance_to_goal'] = float(self.current_observations['distance_to_goal'])
        
        if 'objectgoal' in self.current_observations:
            info['object_goal'] = self.current_observations['objectgoal']
        
        return info

    def evaluate_success(self) -> bool:
        """Evaluate if ObjectNav task was completed successfully"""
        # Check if we're close enough to the target
        if 'distance_to_goal' in self.current_observations:
            distance = float(self.current_observations['distance_to_goal'])
            return distance <= self.success_distance
        
        # Fallback: check if habitat-lab considers it successful
        if hasattr(self.env, 'get_metrics'):
            metrics = self.env.get_metrics()
            return metrics.get('success', False)
        
        return False

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current ObjectNav task"""
        info = {
            'task_type': 'ObjectNav',
            'dataset': self.dataset,
            'target_object': self.target_object,
            'max_steps': self.max_steps,
            'success_distance': self.success_distance,
        }
        
        if self.initial_distance_to_goal != float('inf'):
            info['initial_distance_to_goal'] = self.initial_distance_to_goal
        
        return info

    def get_target_object(self) -> Optional[str]:
        """Get the target object for the current episode"""
        return self.target_object

    def get_distance_to_goal(self) -> Optional[float]:
        """Get current distance to goal if available"""
        if 'distance_to_goal' in self.current_observations:
            return float(self.current_observations['distance_to_goal'])
        return None

    def get_object_goal_sensor(self) -> Optional[Any]:
        """Get object goal sensor data if available"""
        if 'objectgoal' in self.current_observations:
            return self.current_observations['objectgoal']
        return None

    def is_successful(self) -> bool:
        """Check if the current episode is successful"""
        return self.evaluate_success()

    def run_random_episode(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single episode with random actions
        
        Args:
            max_steps: Maximum steps for this episode (overrides default)
            
        Returns:
            Dictionary with episode results
        """
        if max_steps:
            original_max_steps = self.max_steps
            self.max_steps = max_steps
        
        try:
            # Reset for new episode
            self.reset()
            
            while not self._is_episode_done():
                # Choose random action (bias towards moving forward)
                if np.random.random() < 0.7:
                    action = HabitatSimActions.move_forward
                else:
                    action = np.random.choice([
                        HabitatSimActions.turn_left, 
                        HabitatSimActions.turn_right
                    ])
                
                self.step(action)
            
            # Return episode results
            results = self.get_episode_stats()
            results.update(self.get_task_info())
            results['final_distance_to_goal'] = self.get_distance_to_goal()
            
            return results
            
        finally:
            # Restore original max_steps if it was changed
            if max_steps:
                self.max_steps = original_max_steps