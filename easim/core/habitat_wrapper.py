from typing import Dict, Any, Optional
import numpy as np
from easim.utils.habitat_utils import setup_habitat_lab_env
from easim.core.types import Observation, EpisodeInfo, NavigationAction

# Set up habitat environment before importing
setup_habitat_lab_env()

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class HabitatWrapper:
    """Wrapper for Habitat environment with simplified interface."""
    
    def __init__(self, config_path: str):
        """
        Initialize the Habitat environment.
        
        Args:
            config_path: Path to the habitat config file (relative to habitat config dir)
        """
        self.env = habitat.Env(config=habitat.get_config(config_path))
        self.current_episode_info: Optional[EpisodeInfo] = None
        self._setup_actions()
    
    def _setup_actions(self) -> None:
        """Setup available navigation actions."""
        self.actions = {
            'stop': NavigationAction('stop', HabitatSimActions.stop),
            'move_forward': NavigationAction('move_forward', HabitatSimActions.move_forward),
            'turn_left': NavigationAction('turn_left', HabitatSimActions.turn_left),
            'turn_right': NavigationAction('turn_right', HabitatSimActions.turn_right),
        }
    
    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        obs = self.env.reset()
        self._update_episode_info(obs)
        return Observation.from_habitat_obs(obs)
    
    def step(self, action: str) -> tuple[Observation, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action name ('stop', 'move_forward', 'turn_left', 'turn_right')
            
        Returns:
            Tuple of (observation, done, info)
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Available: {list(self.actions.keys())}")
        
        habitat_action = self.actions[action].value
        obs = self.env.step(habitat_action)
        done = self.env.episode_over
        
        info = self._get_episode_metrics()
        self._update_episode_info(obs, done)
        
        return Observation.from_habitat_obs(obs), done, info
    
    def _update_episode_info(self, obs: Dict[str, Any], done: bool = False) -> None:
        """Update current episode information."""
        if self.current_episode_info is None:
            episode = self.env.current_episode
            
            # Handle both objectnav and pointnav goals
            goal = episode.goals[0]
            if hasattr(goal, 'object_category'):
                # ObjectNav goal
                target_object = str(goal.object_category)
            else:
                # PointNav goal
                target_object = "target_position"
            
            self.current_episode_info = EpisodeInfo(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                target_object=target_object,
                target_position=tuple(goal.position),
                start_position=tuple(episode.start_position),
                distance_to_goal=self._get_distance_to_goal(obs)
            )
        
        if done:
            self.current_episode_info.success = self._check_success(obs)
            self.current_episode_info.spl = self._calculate_spl()
    
    def _get_distance_to_goal(self, obs: Dict[str, Any]) -> float:
        """Get distance to goal from observation."""
        # Try different ways to get distance
        if 'distance_to_goal' in obs:
            return float(obs['distance_to_goal'])
        elif 'pointgoal_with_gps_compass' in obs:
            return float(obs['pointgoal_with_gps_compass'][0])
        elif 'objectgoal' in obs:
            # For object nav, we might need to calculate differently
            return 0.0
        return 0.0
    
    def _check_success(self, obs: Dict[str, Any]) -> bool:
        """Check if episode was successful."""
        # Check if we're close enough to the goal
        distance = self._get_distance_to_goal(obs)
        return distance < 0.1  # Success threshold
    
    def _calculate_spl(self) -> float:
        """Calculate Success weighted by Path Length (SPL)."""
        if not self.current_episode_info or not self.current_episode_info.success:
            return 0.0
        
        optimal_distance = np.linalg.norm(
            np.array(self.current_episode_info.target_position) - 
            np.array(self.current_episode_info.start_position)
        )
        
        if self.current_episode_info.steps == 0:
            return 0.0
        
        return optimal_distance / max(optimal_distance, self.current_episode_info.steps * 0.25)  # 0.25m per step
    
    def _get_episode_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics."""
        if not self.current_episode_info:
            return {}
        
        return {
            'episode_id': self.current_episode_info.episode_id,
            'scene_id': self.current_episode_info.scene_id,
            'target_object': self.current_episode_info.target_object,
            'steps': self.current_episode_info.steps,
            'distance_to_goal': self.current_episode_info.distance_to_goal
        }
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    @property
    def episode_over(self) -> bool:
        """Check if current episode is over."""
        return self.env.episode_over