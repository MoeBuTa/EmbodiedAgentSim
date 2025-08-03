"""Point Navigation task implementation for habitat-lab framework"""
from typing import Dict, Any, Optional
import numpy as np

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

from habitat.sims.habitat_simulator.actions import HabitatSimActions, HabitatSimActionsSingleton

from .base import BaseEmbodiedTask


class PointNavTask(BaseEmbodiedTask):
    """Point Navigation task using habitat-lab framework"""

    def __init__(self, dataset: str = "MP3D", **kwargs):
        """
        Initialize PointNav task
        
        Args:
            dataset: Dataset to use ("MP3D" or "HM3D")
            **kwargs: Additional arguments passed to base class
        """
        # Choose config based on dataset
        if dataset.upper() == "MP3D":
            config_path = "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
        elif dataset.upper() == "HM3D":
            config_path = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        self.dataset = dataset
        self.target_position = None
        self.initial_distance_to_goal = None
        
        # PointNav specific parameters
        kwargs.setdefault('max_steps', 500)
        kwargs.setdefault('success_distance', 0.2)  # Slightly larger threshold for point nav
        
        super().__init__(config_path, **kwargs)

    def _on_reset(self):
        """PointNav-specific reset logic"""
        # Get target position information if available
        if hasattr(self.env.current_episode, 'goals'):
            goals = self.env.current_episode.goals
            if goals:
                self.target_position = goals[0].position
        
        # Get initial distance to goal from pointgoal_with_gps_compass sensor
        if 'pointgoal_with_gps_compass' in self.current_observations:
            self.initial_distance_to_goal = float(self.current_observations['pointgoal_with_gps_compass'][0])
        else:
            self.initial_distance_to_goal = float('inf')
        
        print(f"New PointNav episode started")
        if self.target_position is not None:
            print(f"Target position: {self.target_position}")
        if self.initial_distance_to_goal != float('inf'):
            print(f"Initial distance to goal: {self.initial_distance_to_goal:.2f}m")

    def _process_action(self, action: Any) -> Any:
        """Process and validate action for PointNav"""
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
        """Check if PointNav episode should end"""
        # Episode is done if habitat-lab says so or max steps reached
        return self.is_episode_over or self.step_count >= self.max_steps

    def _update_episode_stats(self):
        """Update PointNav-specific episode statistics"""
        # Update basic stats
        self.episode_stats['steps'] = self.step_count
        
        # Update distance to goal if available
        if 'pointgoal_with_gps_compass' in self.current_observations:
            self.episode_stats['distance_to_goal'] = float(self.current_observations['pointgoal_with_gps_compass'][0])
        
        # Check for success
        self.episode_stats['success'] = self.evaluate_success()
        
        # Calculate path efficiency if we have initial distance
        if (self.initial_distance_to_goal != float('inf') and 
            self.initial_distance_to_goal > 0):
            current_distance = self.episode_stats.get('distance_to_goal', float('inf'))
            if current_distance != float('inf'):
                self.episode_stats['path_efficiency'] = self.initial_distance_to_goal / max(current_distance, 0.01)

    def _get_step_info(self) -> Dict[str, Any]:
        """Get PointNav-specific step information"""
        info = {
            'step': self.step_count,
            'task_type': 'PointNav',
            'episode_over': self.is_episode_over,
        }
        
        # Add goal information if available
        if 'pointgoal_with_gps_compass' in self.current_observations:
            compass_data = self.current_observations['pointgoal_with_gps_compass']
            info['distance_to_goal'] = float(compass_data[0])
            info['angle_to_goal'] = float(compass_data[1])
        
        if self.target_position is not None:
            info['target_position'] = self.target_position
        
        return info

    def evaluate_success(self) -> bool:
        """Evaluate if PointNav task was completed successfully"""
        # Check if we're close enough to the target
        if 'pointgoal_with_gps_compass' in self.current_observations:
            distance = float(self.current_observations['pointgoal_with_gps_compass'][0])
            return distance <= self.success_distance
        
        # Fallback: check if habitat-lab considers it successful
        if hasattr(self.env, 'get_metrics'):
            metrics = self.env.get_metrics()
            return metrics.get('success', False)
        
        return False

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current PointNav task"""
        info = {
            'task_type': 'PointNav',
            'dataset': self.dataset,
            'max_steps': self.max_steps,
            'success_distance': self.success_distance,
        }
        
        if self.target_position is not None:
            info['target_position'] = self.target_position
        
        if self.initial_distance_to_goal != float('inf'):
            info['initial_distance_to_goal'] = self.initial_distance_to_goal
        
        return info

    def get_distance_to_goal(self) -> Optional[float]:
        """Get current distance to goal if available"""
        if 'pointgoal_with_gps_compass' in self.current_observations:
            return float(self.current_observations['pointgoal_with_gps_compass'][0])
        return None

    def get_angle_to_goal(self) -> Optional[float]:
        """Get current angle to goal if available"""
        if 'pointgoal_with_gps_compass' in self.current_observations:
            return float(self.current_observations['pointgoal_with_gps_compass'][1])
        return None

    def get_gps_compass(self) -> Optional[np.ndarray]:
        """Get GPS compass sensor data if available"""
        if 'pointgoal_with_gps_compass' in self.current_observations:
            return self.current_observations['pointgoal_with_gps_compass']
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
            results['final_angle_to_goal'] = self.get_angle_to_goal()
            
            return results
            
        finally:
            # Restore original max_steps if it was changed
            if max_steps:
                self.max_steps = original_max_steps