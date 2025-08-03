"""Base embodied task class inheriting from habitat-lab EmbodiedTask"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import numpy as np

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.embodied_task import EmbodiedTask

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseEmbodiedTask(ABC):
    """
    Simplified base class for embodied AI tasks using habitat-lab framework.
    
    This wraps habitat-lab's Env for easier usage while maintaining compatibility.
    For advanced usage, inherit directly from habitat.core.embodied_task.EmbodiedTask.
    """

    def __init__(self, config_path: str, **kwargs):
        """
        Initialize the task with a habitat-lab config
        
        Args:
            config_path: Path to habitat-lab config file
            **kwargs: Additional arguments for task customization
        """
        self.config_path = config_path
        self.env = None
        self.current_observations = None
        self.episode_stats = {}
        self.step_count = 0
        
        # Task-specific parameters
        self.max_steps = kwargs.get('max_steps', 500)
        self.success_distance = kwargs.get('success_distance', 0.2)
        
        self._setup_environment()

    def _setup_environment(self):
        """Setup the habitat-lab environment"""
        try:
            config = habitat.get_config(self.config_path)
            self.env = habitat.Env(config=config)
            print(f"Environment created successfully with config: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to create environment: {e}")

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the task for a new episode"""
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        
        self.current_observations = self.env.reset()
        self.step_count = 0
        self.episode_stats = {
            'success': False,
            'steps': 0,
            'distance_to_goal': float('inf'),
            'path_length': 0.0
        }
        
        self._on_reset()
        return self.current_observations

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Action to take (can be action enum or processed action)
            
        Returns:
            Tuple of (observations, done, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        
        # Convert action if needed
        processed_action = self._process_action(action)
        
        # Take step
        self.current_observations = self.env.step(processed_action)
        self.step_count += 1
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Update episode stats
        self._update_episode_stats()
        
        # Get additional info
        info = self._get_step_info()
        
        return self.current_observations, done, info

    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def is_episode_over(self) -> bool:
        """Check if current episode is over"""
        return self.env.episode_over if self.env else True

    @property
    def available_actions(self) -> List[Any]:
        """Get list of available actions"""
        return [
            HabitatSimActions.move_forward,
            HabitatSimActions.turn_left,
            HabitatSimActions.turn_right,
            HabitatSimActions.stop
        ]

    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def _on_reset(self):
        """Called after environment reset, implement task-specific reset logic"""
        pass

    @abstractmethod
    def _process_action(self, action: Any) -> Any:
        """Process and validate action before sending to environment"""
        pass

    @abstractmethod
    def _is_episode_done(self) -> bool:
        """Check if episode should end based on task-specific criteria"""
        pass

    @abstractmethod
    def _update_episode_stats(self):
        """Update task-specific episode statistics"""
        pass

    @abstractmethod
    def _get_step_info(self) -> Dict[str, Any]:
        """Get additional information about the current step"""
        pass

    @abstractmethod
    def evaluate_success(self) -> bool:
        """Evaluate if the task was completed successfully"""
        pass

    @abstractmethod
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current task/goal"""
        pass

    # Utility methods

    def get_observations(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current observations"""
        return self.current_observations

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics"""
        return self.episode_stats.copy()

    def get_rgb_observation(self) -> Optional[np.ndarray]:
        """Get RGB observation if available"""
        if self.current_observations and 'rgb' in self.current_observations:
            return self.current_observations['rgb']
        return None

    def get_depth_observation(self) -> Optional[np.ndarray]:
        """Get depth observation if available"""
        if self.current_observations and 'depth' in self.current_observations:
            return self.current_observations['depth']
        return None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class HabitatEmbodiedTask(EmbodiedTask):
    """
    Advanced base class that directly inherits from habitat-lab's EmbodiedTask.
    
    Use this for advanced scenarios where you need full habitat-lab functionality
    like custom actions, measures, and sensors.
    """
    
    def __init__(self, config: "DictConfig", sim, dataset=None):
        super().__init__(config, sim, dataset)
        
    def _check_episode_is_active(self, *args, action, episode, **kwargs) -> bool:
        """
        Default implementation - override in subclasses for custom termination logic
        """
        # Basic implementation - episode ends when max steps reached
        if hasattr(self, '_max_episode_steps'):
            return self._step_count < self._max_episode_steps
        return True  # Episode continues by default
        
    @abstractmethod
    def get_task_observations(self) -> Dict[str, Any]:
        """Get task-specific observations"""
        pass
        
    @abstractmethod 
    def calculate_task_reward(self) -> float:
        """Calculate task-specific reward"""
        pass