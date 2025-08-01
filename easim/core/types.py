from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class NavigationAction:
    """Represents a navigation action with its name and value."""
    name: str
    value: int


@dataclass
class Observation:
    """Represents an observation from the environment."""
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    objectgoal: Optional[np.ndarray] = None
    gps: Optional[np.ndarray] = None
    compass: Optional[np.ndarray] = None
    
    @classmethod
    def from_habitat_obs(cls, obs: Dict[str, Any]) -> 'Observation':
        """Create Observation from habitat observation dict."""
        return cls(
            rgb=obs.get('rgb'),
            depth=obs.get('depth'),
            objectgoal=obs.get('objectgoal'),
            gps=obs.get('gps'),
            compass=obs.get('compass')
        )


@dataclass
class EpisodeInfo:
    """Information about the current episode."""
    episode_id: str
    scene_id: str
    target_object: str
    target_position: Tuple[float, float, float]
    start_position: Tuple[float, float, float]
    distance_to_goal: float
    success: bool = False
    spl: float = 0.0
    steps: int = 0