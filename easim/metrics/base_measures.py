"""Base measure classes extending habitat-lab's Measure for custom metrics"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

from habitat.core.embodied_task import Measure


class BaseMeasure(Measure):
    """
    Base class for custom measures extending habitat-lab's Measure
    
    Use this for creating task-specific metrics that need to be tracked
    during episodes and collected for analysis.
    """
    
    def __init__(self, sim, config, task, *args, **kwargs):
        """
        Initialize base measure
        
        Args:
            sim: Simulator instance
            config: Configuration for the measure
            task: Task instance
        """
        super().__init__()
        self._sim = sim
        self._config = config
        self._task = task
        self.uuid = self._get_uuid()
        self._metric = None

    @abstractmethod
    def _get_uuid(self, *args, **kwargs) -> str:
        """Return unique identifier for this measure"""
        pass

    @abstractmethod 
    def reset_metric(self, *args, **kwargs):
        """Reset the metric for a new episode"""
        pass

    @abstractmethod
    def update_metric(self, *args, **kwargs):
        """Update the metric during episode step"""
        pass


class DistanceToGoalMeasure(BaseMeasure):
    """Generic distance to goal measure"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "distance_to_goal"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset distance tracking for new episode"""
        self._metric = float('inf')
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update distance from current observations"""
        # Try different distance sensors
        if 'distance_to_goal' in observations:
            self._metric = float(observations['distance_to_goal'])
        elif 'pointgoal_with_gps_compass' in observations:
            self._metric = float(observations['pointgoal_with_gps_compass'][0])
        else:
            self._metric = float('inf')


class SuccessMeasure(BaseMeasure):
    """Generic success measure"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._success_distance = getattr(config, 'success_distance', 0.2)
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "success"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset success for new episode"""
        self._metric = False
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update success based on distance to goal"""
        # Get current distance
        distance = float('inf')
        if 'distance_to_goal' in observations:
            distance = float(observations['distance_to_goal'])
        elif 'pointgoal_with_gps_compass' in observations:
            distance = float(observations['pointgoal_with_gps_compass'][0])
        
        # Check success condition
        self._metric = distance <= self._success_distance


class EpisodeStepsMeasure(BaseMeasure):
    """Count steps taken in episode"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "episode_steps"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset step counter for new episode"""
        self._metric = 0
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Increment step counter"""
        self._metric += 1


class PathLengthMeasure(BaseMeasure):
    """Measure total path length traveled"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._previous_position = None
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "path_length"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset path length tracking"""
        self._metric = 0.0
        self._previous_position = None
        
        # Get initial position
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._previous_position = agent_state.position
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update path length by calculating distance moved"""
        if self._sim and self._previous_position is not None:
            current_position = self._sim.get_agent_state().position
            
            # Calculate distance moved
            distance_moved = np.linalg.norm(current_position - self._previous_position)
            self._metric += distance_moved
            
            # Update previous position
            self._previous_position = current_position


class PathEfficiencyMeasure(BaseMeasure):
    """Measure path efficiency (straight-line distance / actual path length)"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._start_position = None
        self._goal_position = None
        self._path_length = 0.0
        self._previous_position = None
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "path_efficiency"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset efficiency tracking"""
        self._metric = 0.0
        self._path_length = 0.0
        self._previous_position = None
        
        # Get start position
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._start_position = agent_state.position.copy()
            self._previous_position = agent_state.position.copy()
        
        # Get goal position if available
        if hasattr(episode, 'goals') and episode.goals:
            self._goal_position = np.array(episode.goals[0].position)
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update path efficiency calculation"""
        if self._sim and self._previous_position is not None:
            current_position = self._sim.get_agent_state().position
            
            # Update path length
            distance_moved = np.linalg.norm(current_position - self._previous_position)
            self._path_length += distance_moved
            self._previous_position = current_position
            
            # Calculate efficiency if we have goal position
            if (self._goal_position is not None and 
                self._start_position is not None and 
                self._path_length > 0):
                
                straight_line_distance = np.linalg.norm(self._goal_position - self._start_position)
                self._metric = straight_line_distance / self._path_length
            else:
                self._metric = 0.0


class TimeToGoalMeasure(BaseMeasure):
    """Measure time taken to reach goal"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._start_time = None
        self._goal_reached = False
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "time_to_goal"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset time tracking"""
        import time
        self._start_time = time.time()
        self._goal_reached = False
        self._metric = 0.0
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update time if goal is reached"""
        if not self._goal_reached:
            # Check if goal is reached
            distance = float('inf')
            if 'distance_to_goal' in observations:
                distance = float(observations['distance_to_goal'])
            elif 'pointgoal_with_gps_compass' in observations:
                distance = float(observations['pointgoal_with_gps_compass'][0])
            
            success_distance = getattr(self._config, 'success_distance', 0.2)
            if distance <= success_distance:
                import time
                self._metric = time.time() - self._start_time
                self._goal_reached = True


class CollisionCountMeasure(BaseMeasure):
    """Count number of collisions during episode"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._previous_position = None
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "collision_count"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset collision counter"""
        self._metric = 0
        self._previous_position = None
        
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._previous_position = agent_state.position
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Detect collisions by checking if position didn't change on forward action"""
        if (self._sim and self._previous_position is not None and 
            hasattr(action, 'get') and action.get('action') == 'move_forward'):
            
            current_position = self._sim.get_agent_state().position
            distance_moved = np.linalg.norm(current_position - self._previous_position)
            
            # If we tried to move forward but didn't move much, it's likely a collision
            if distance_moved < 0.01:  # Small threshold for floating point precision
                self._metric += 1
            
            self._previous_position = current_position