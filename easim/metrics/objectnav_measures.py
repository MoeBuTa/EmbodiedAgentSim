"""ObjectNav-specific measures extending base measures"""
from typing import Any, Dict, Optional
import numpy as np

from .base_measures import BaseMeasure, SuccessMeasure, DistanceToGoalMeasure


class ObjectNavSuccessMeasure(SuccessMeasure):
    """Success measure specifically for ObjectNav tasks"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "objectnav_success"
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """ObjectNav-specific success logic"""
        # Use ObjectNav-specific distance sensor
        if 'distance_to_goal' in observations:
            distance = float(observations['distance_to_goal'])
            self._metric = distance <= self._success_distance
        else:
            # Fallback to task evaluation if available
            task = kwargs.get('task')
            if task and hasattr(task, 'evaluate_success'):
                self._metric = task.evaluate_success()
            else:
                self._metric = False


class ObjectNavDistanceMeasure(DistanceToGoalMeasure):
    """Distance measure specifically for ObjectNav"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "objectnav_distance_to_goal"
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """ObjectNav-specific distance tracking"""
        if 'distance_to_goal' in observations:
            self._metric = float(observations['distance_to_goal'])
        else:
            self._metric = float('inf')


class ObjectFoundMeasure(BaseMeasure):
    """Track if target object is visible/found"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._target_object = None
        self._object_found_steps = []
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "object_found"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset object finding tracking"""
        self._metric = False
        self._object_found_steps = []
        
        # Get target object from episode
        if hasattr(episode, 'object_category'):
            self._target_object = episode.object_category
        else:
            self._target_object = None
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Check if target object is visible in current observation"""
        # This is a simplified version - in practice you'd use semantic segmentation
        # or object detection to determine if the target object is visible
        
        if 'objectgoal' in observations:
            # Object goal sensor indicates target object category
            self._target_object = observations['objectgoal']
        
        # Check distance - if very close, assume object is found
        if 'distance_to_goal' in observations:
            distance = float(observations['distance_to_goal'])
            if distance <= 1.0:  # Within 1 meter, assume object is visible
                if not self._metric:  # First time finding it
                    task = kwargs.get('task')
                    step_count = getattr(task, 'step_count', 0) if task else 0
                    self._object_found_steps.append(step_count)
                self._metric = True


class ObjectCategoryMeasure(BaseMeasure):
    """Track the target object category for the episode"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "target_object_category"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Get target object category for new episode"""
        if hasattr(episode, 'object_category'):
            self._metric = episode.object_category
        else:
            self._metric = "unknown"
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Object category doesn't change during episode"""
        # Update from observations if available
        if 'objectgoal' in observations and self._metric == "unknown":
            self._metric = observations['objectgoal']


class ObjectSearchEfficiencyMeasure(BaseMeasure):
    """Measure efficiency of object search (distance traveled vs. optimal path)"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._start_position = None
        self._object_position = None
        self._path_length = 0.0
        self._previous_position = None
        self._object_found = False
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "object_search_efficiency"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset search efficiency tracking"""
        self._metric = 0.0
        self._path_length = 0.0
        self._object_found = False
        self._previous_position = None
        
        # Get start position
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._start_position = agent_state.position.copy()
            self._previous_position = agent_state.position.copy()
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update search efficiency calculation"""
        if self._sim and self._previous_position is not None:
            current_position = self._sim.get_agent_state().position
            
            # Update path length
            distance_moved = np.linalg.norm(current_position - self._previous_position)
            self._path_length += distance_moved
            self._previous_position = current_position
            
            # Check if object is found
            if not self._object_found and 'distance_to_goal' in observations:
                distance_to_goal = float(observations['distance_to_goal'])
                if distance_to_goal <= 1.0:  # Object found
                    self._object_found = True
                    self._object_position = current_position.copy()
                    
                    # Calculate efficiency
                    if self._start_position is not None:
                        optimal_distance = np.linalg.norm(self._object_position - self._start_position)
                        if self._path_length > 0:
                            self._metric = optimal_distance / self._path_length


class ObjectViewTimeMeasure(BaseMeasure):
    """Track how long the target object has been in view"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._view_start_step = None
        self._total_view_time = 0
        self._currently_viewing = False
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "object_view_time"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset view time tracking"""
        self._metric = 0
        self._view_start_step = None
        self._total_view_time = 0
        self._currently_viewing = False
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update object view time"""
        # Check if object is currently visible (simplified - distance based)
        object_visible = False
        if 'distance_to_goal' in observations:
            distance = float(observations['distance_to_goal'])
            object_visible = distance <= 2.0  # Within 2 meters
        
        task = kwargs.get('task')
        current_step = getattr(task, 'step_count', 0) if task else 0
        
        if object_visible and not self._currently_viewing:
            # Object just came into view
            self._currently_viewing = True
            self._view_start_step = current_step
        elif not object_visible and self._currently_viewing:
            # Object just went out of view
            self._currently_viewing = False
            if self._view_start_step is not None:
                self._total_view_time += (current_step - self._view_start_step)
        
        # Update metric with total view time
        if self._currently_viewing and self._view_start_step is not None:
            self._metric = self._total_view_time + (current_step - self._view_start_step)
        else:
            self._metric = self._total_view_time