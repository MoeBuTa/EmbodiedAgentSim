"""
Base configuration dataclasses for tasks
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class TaskConfig:
    """Base task configuration"""
    max_episode_steps: int = 500
    success_distance: float = 0.5
    step_penalty: float = -0.01
    success_reward: float = 10.0
    collision_penalty: float = -0.1
    out_of_bounds_penalty: float = -1.0
    action_space_type: str = "discrete_nav"
    sensor_suite_type: str = "full_vision"

    # Episode sampling
    shuffle_episodes: bool = True
    max_episodes: Optional[int] = None

    # Evaluation
    split: str = "val"

    # Additional task-specific parameters
    task_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass


@dataclass
class NavigationConfig(TaskConfig):
    """Navigation task configuration"""
    # Distance-based reward shaping
    distance_reward_scale: float = 0.1
    enable_distance_reward: bool = False

    # Navigation specific penalties
    collision_threshold: float = 0.1
    stuck_threshold: int = 10  # Steps without progress


@dataclass
class SimulatorConfig:
    """Simulator configuration"""
    scene_path: Optional[str] = None
    scene_dataset: Optional[str] = None
    dataset_type: str = "MP3D"  # MP3D, HM3D, or custom
    gpu_device_id: int = 0
    enable_physics: bool = False
    random_seed: Optional[int] = None
    allow_sliding: bool = True
    frustum_culling: bool = True


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_id: int = 0
    height: float = 1.5
    radius: float = 0.2
    mass: float = 32.0
    linear_acceleration: float = 20.0
    angular_acceleration: float = 4.0
    linear_friction: float = 0.5
    angular_friction: float = 1.0
    coefficient_of_restitution: float = 0.0
    sensor_height: float = 1.25
    action_space_type: str = "discrete_nav"
    sensor_suite_type: str = "full_vision"
    action_space_config: Optional[Dict[str, Any]] = None
    sensor_suite_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.action_space_config is None:
            self.action_space_config = {}
        if self.sensor_suite_config is None:
            self.sensor_suite_config = {}


@dataclass
class SensorConfig:
    """Sensor configuration"""
    uuid: str
    sensor_type: str
    resolution: tuple = (256, 256)
    position: List[float] = field(default_factory=lambda: [0.0, 1.25, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ActionSpec:
    """Action specification"""
    name: str
    action_type: str
    actuation_spec: Optional[Any] = None
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}