"""PointNav-specific measures extending base measures"""
from typing import Any, Dict, Optional
import numpy as np

from .base_measures import BaseMeasure, SuccessMeasure, DistanceToGoalMeasure


class PointNavSuccessMeasure(SuccessMeasure):
    """Success measure specifically for PointNav tasks"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "pointnav_success"
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """PointNav-specific success logic"""
        # Use PointNav GPS compass sensor
        if 'pointgoal_with_gps_compass' in observations:
            distance = float(observations['pointgoal_with_gps_compass'][0])
            self._metric = distance <= self._success_distance
        else:
            # Fallback to task evaluation if available
            task = kwargs.get('task')
            if task and hasattr(task, 'evaluate_success'):
                self._metric = task.evaluate_success()
            else:
                self._metric = False


class PointNavDistanceMeasure(DistanceToGoalMeasure):
    """Distance measure specifically for PointNav"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "pointnav_distance_to_goal"
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """PointNav-specific distance tracking"""
        if 'pointgoal_with_gps_compass' in observations:
            self._metric = float(observations['pointgoal_with_gps_compass'][0])
        else:
            self._metric = float('inf')


class AngleToGoalMeasure(BaseMeasure):
    """Track angle to goal for PointNav tasks"""
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "angle_to_goal"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset angle tracking"""
        self._metric = 0.0
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update angle to goal from GPS compass sensor"""
        if 'pointgoal_with_gps_compass' in observations:
            self._metric = float(observations['pointgoal_with_gps_compass'][1])
        else:
            self._metric = 0.0


class NavigationErrorMeasure(BaseMeasure):
    """Track cumulative navigation error (deviation from optimal path)"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._start_position = None
        self._goal_position = None
        self._cumulative_error = 0.0
        self._previous_position = None
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "navigation_error"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset navigation error tracking"""
        self._metric = 0.0
        self._cumulative_error = 0.0
        self._previous_position = None
        
        # Get start position
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._start_position = agent_state.position.copy()
            self._previous_position = agent_state.position.copy()
        
        # Get goal position
        if hasattr(episode, 'goals') and episode.goals:
            self._goal_position = np.array(episode.goals[0].position)
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Calculate navigation error as deviation from optimal path"""
        if (self._sim and self._previous_position is not None and 
            self._goal_position is not None and self._start_position is not None):
            
            current_position = self._sim.get_agent_state().position
            
            # Calculate optimal path direction
            optimal_direction = self._goal_position - self._start_position
            optimal_direction = optimal_direction / np.linalg.norm(optimal_direction)
            
            # Calculate actual movement direction
            movement = current_position - self._previous_position
            movement_distance = np.linalg.norm(movement)
            
            if movement_distance > 0.01:  # Only if we actually moved
                movement_direction = movement / movement_distance
                
                # Calculate angle between optimal and actual direction
                dot_product = np.dot(optimal_direction, movement_direction)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle_error = np.arccos(dot_product)
                
                # Add to cumulative error (weighted by distance moved)
                self._cumulative_error += angle_error * movement_distance
            
            self._previous_position = current_position
            self._metric = self._cumulative_error


class PointNavTimeToGoalMeasure(BaseMeasure):
    """Time taken to reach the goal in PointNav"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._start_time = None
        self._goal_reached = False
        self._success_distance = getattr(config, 'success_distance', 0.2)
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "pointnav_time_to_goal"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset time tracking"""
        import time
        self._start_time = time.time()
        self._goal_reached = False
        self._metric = 0.0
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Update time when goal is reached"""
        if not self._goal_reached and 'pointgoal_with_gps_compass' in observations:
            distance = float(observations['pointgoal_with_gps_compass'][0])
            
            if distance <= self._success_distance:
                import time
                self._metric = time.time() - self._start_time
                self._goal_reached = True


class TurningAngleMeasure(BaseMeasure):
    """Track total turning angle during navigation"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._previous_rotation = None
        self._total_turning = 0.0
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "total_turning_angle"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset turning tracking"""
        self._metric = 0.0
        self._total_turning = 0.0
        self._previous_rotation = None
        
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._previous_rotation = agent_state.rotation
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Track turning angle between steps"""
        if self._sim and self._previous_rotation is not None:
            current_rotation = self._sim.get_agent_state().rotation
            
            # Calculate rotation difference
            # This is a simplified calculation - in practice you'd use quaternion math
            rotation_diff = abs(current_rotation - self._previous_rotation)
            
            # Accumulate turning
            self._total_turning += float(rotation_diff)
            self._metric = self._total_turning
            
            self._previous_rotation = current_rotation


class StraightLineDeviationMeasure(BaseMeasure):
    """Measure how much the agent deviates from straight line to goal"""
    
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim, config, task, *args, **kwargs)
        self._start_position = None
        self._goal_position = None
        self._max_deviation = 0.0
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return "straight_line_deviation"
    
    def reset_metric(self, episode, *args, **kwargs):
        """Reset deviation tracking"""
        self._metric = 0.0
        self._max_deviation = 0.0
        
        # Get start and goal positions
        if self._sim:
            agent_state = self._sim.get_agent_state()
            self._start_position = agent_state.position.copy()
        
        if hasattr(episode, 'goals') and episode.goals:
            self._goal_position = np.array(episode.goals[0].position)
    
    def update_metric(self, episode, action, observations, *args, **kwargs):
        """Calculate deviation from straight line path"""
        if (self._sim and self._start_position is not None and 
            self._goal_position is not None):
            
            current_position = self._sim.get_agent_state().position
            
            # Calculate distance from current position to the straight line
            # between start and goal
            line_vector = self._goal_position - self._start_position
            line_length = np.linalg.norm(line_vector)
            
            if line_length > 0:
                # Project current position onto the line
                t = np.dot(current_position - self._start_position, line_vector) / (line_length ** 2)
                t = np.clip(t, 0, 1)  # Clamp to line segment
                
                closest_point = self._start_position + t * line_vector
                deviation = np.linalg.norm(current_position - closest_point)
                
                # Track maximum deviation
                self._max_deviation = max(self._max_deviation, deviation)
                self._metric = self._max_deviation