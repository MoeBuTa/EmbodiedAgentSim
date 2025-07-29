"""
Clean wrapper interfaces for Habitat Lab integration
"""
import habitat
from habitat.config.default import get_config
from habitat.core.env import Env
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from easim.utils.constants import (
    MP3D_SCENE_DATASET, HM3D_SCENE_DATASET,
    DEFAULT_FORWARD_STEP, DEFAULT_TURN_ANGLE,
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION
)
from easim.core.video_recorder import (
    VideoRecorder, BaseNavigationStrategy, PygameVisualizer
)


class TaskType(Enum):
    """Supported task types"""
    POINTNAV = "pointnav"
    OBJECTNAV = "objectnav"
    IMAGENAV = "imagenav"
    VLN = "vln"
    EQA = "eqa"


class DatasetType(Enum):
    """Supported dataset types"""
    MP3D = "mp3d"
    HM3D = "hm3d"
    GIBSON = "gibson"


@dataclass
class HabitatWrapperConfig:
    """Configuration for Habitat wrapper"""
    task_type: TaskType = TaskType.POINTNAV
    dataset_type: DatasetType = DatasetType.MP3D
    scene_dataset_path: Optional[str] = None
    split: str = "train"
    
    # Simulator settings
    width: int = 256
    height: int = 256
    forward_step_size: float = DEFAULT_FORWARD_STEP
    turn_angle: int = int(DEFAULT_TURN_ANGLE)
    
    # Agent settings
    sensor_height: float = 1.25
    enable_physics: bool = False
    seed: Optional[int] = None
    
    # Episode settings
    max_episode_length: int = 500
    

class HabitatEnvironmentWrapper:
    """Clean wrapper around Habitat Lab environment with video recording support"""
    
    def __init__(self, config: HabitatWrapperConfig):
        self.config = config
        self.env: Optional[Env] = None
        self._current_episode = 0
        self._episode_stats = {}
        
        # Video recording support
        self._video_recorder: Optional[VideoRecorder] = None
        self._recording = False
        
        # Initialize environment
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup Habitat environment with configuration"""
        # Get base config path
        config_path = self._get_config_path()
        
        # Load Habitat config
        habitat_config = get_config(config_path)
        
        # Override with our settings
        self._configure_habitat_settings(habitat_config)
        
        # Create environment
        self.env = habitat.Env(config=habitat_config)
        
        print(f"Habitat environment created:")
        print(f"  Task: {self.config.task_type.value}")
        print(f"  Dataset: {self.config.dataset_type.value}")
        print(f"  Action space: {self.env.action_space}")
        print(f"  Observation space keys: {list(self.env.observation_space.spaces.keys())}")
    
    def _get_config_path(self) -> str:
        """Get appropriate config path based on task and dataset"""
        base_path = "habitat-lab/habitat/config/benchmark/nav"
        
        if self.config.task_type == TaskType.POINTNAV:
            if self.config.dataset_type == DatasetType.MP3D:
                return f"{base_path}/pointnav/pointnav_mp3d.yaml"
            elif self.config.dataset_type == DatasetType.HM3D:
                return f"{base_path}/pointnav/pointnav_hm3d.yaml"
        elif self.config.task_type == TaskType.OBJECTNAV:
            if self.config.dataset_type == DatasetType.MP3D:
                return f"{base_path}/objectnav/objectnav_mp3d.yaml"
            elif self.config.dataset_type == DatasetType.HM3D:
                return f"{base_path}/objectnav/objectnav_hm3d.yaml"
        
        # Default fallback
        return f"{base_path}/pointnav/pointnav_mp3d.yaml"
    
    def _configure_habitat_settings(self, habitat_config):
        """Configure Habitat settings based on our config"""
        # Simulator settings
        habitat_config.habitat.simulator.forward_step_size = self.config.forward_step_size
        habitat_config.habitat.simulator.turn_angle = self.config.turn_angle
        habitat_config.habitat.simulator.rgb_sensor.width = self.config.width
        habitat_config.habitat.simulator.rgb_sensor.height = self.config.height
        habitat_config.habitat.simulator.depth_sensor.width = self.config.width
        habitat_config.habitat.simulator.depth_sensor.height = self.config.height
        
        # Task settings
        habitat_config.habitat.task.max_episode_steps = self.config.max_episode_length
        
        # Dataset settings
        if self.config.scene_dataset_path:
            habitat_config.habitat.simulator.scene_dataset = self.config.scene_dataset_path
        elif self.config.dataset_type == DatasetType.MP3D:
            habitat_config.habitat.simulator.scene_dataset = str(MP3D_SCENE_DATASET)
        elif self.config.dataset_type == DatasetType.HM3D:
            habitat_config.habitat.simulator.scene_dataset = str(HM3D_SCENE_DATASET)
        
        # Seed
        if self.config.seed is not None:
            habitat_config.habitat.seed = self.config.seed
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and start new episode"""
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        
        observations = self.env.reset()
        self._current_episode += 1
        self._episode_stats = {
            'episode_id': self._current_episode,
            'steps': 0,
            'total_reward': 0.0,
            'success': False
        }
        
        return self._process_observations(observations)
    
    def step(self, action: Union[int, str, Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        
        # Convert action if needed
        action = self._process_action(action)
        
        # Take step
        observations = self.env.step(action)
        
        # Get episode info
        episode_over = self.env.episode_over
        info = self.env.get_metrics()
        
        # Calculate reward
        reward = self._calculate_reward(info)
        
        # Update episode stats
        self._episode_stats['steps'] += 1
        self._episode_stats['total_reward'] += reward
        if episode_over and info.get('success', 0.0) > 0.5:
            self._episode_stats['success'] = True
        
        # Record frame if recording
        processed_obs = self._process_observations(observations)
        if self._recording and self._video_recorder and 'rgb' in processed_obs:
            self._video_recorder.add_frame(processed_obs['rgb'])
        
        return processed_obs, reward, episode_over, info
    
    def _process_action(self, action: Union[int, str, Dict[str, Any]]) -> int:
        """Process action input to environment-compatible format"""
        if isinstance(action, str):
            # Convert string action to integer
            action_mapping = {
                'stop': 0,
                'move_forward': 1,
                'turn_left': 2,
                'turn_right': 3,
                'look_up': 4,
                'look_down': 5
            }
            return action_mapping.get(action, 0)
        elif isinstance(action, dict):
            # Handle structured actions
            return action.get('action', 0)
        else:
            return int(action)
    
    def _process_observations(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process raw observations into standardized format"""
        processed = {}
        
        # Standard sensors
        if 'rgb' in observations:
            processed['rgb'] = observations['rgb']
        if 'depth' in observations:
            processed['depth'] = observations['depth']
        if 'semantic' in observations:
            processed['semantic'] = observations['semantic']
        
        # Navigation-specific sensors
        if 'pointgoal_with_gps_compass' in observations:
            processed['pointgoal'] = observations['pointgoal_with_gps_compass']
        if 'objectgoal' in observations:
            processed['objectgoal'] = observations['objectgoal']
        if 'compass' in observations:
            processed['compass'] = observations['compass']
        if 'gps' in observations:
            processed['gps'] = observations['gps']
        
        return processed
    
    def _calculate_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward based on metrics"""
        # Use distance_to_goal_reward if available
        if 'distance_to_goal_reward' in info:
            return info['distance_to_goal_reward']
        
        # Fallback reward calculation
        reward = 0.0
        if 'success' in info and info['success'] > 0.5:
            reward += 10.0  # Success bonus
        
        if 'distance_to_goal' in info:
            # Small penalty for distance (encourage getting closer)
            reward -= info['distance_to_goal'] * 0.01
        
        return reward
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics"""
        return self._episode_stats.copy()
    
    def get_scene_info(self) -> Dict[str, Any]:
        """Get current scene information"""
        if self.env is None or not hasattr(self.env, '_sim'):
            return {}
        
        scene_info = {
            'scene_id': getattr(self.env.current_episode, 'scene_id', 'unknown'),
            'episode_id': getattr(self.env.current_episode, 'episode_id', 'unknown'),
        }
        
        # Add goal information if available
        if hasattr(self.env.current_episode, 'goals') and self.env.current_episode.goals:
            goal = self.env.current_episode.goals[0]
            scene_info['goal_position'] = getattr(goal, 'position', None)
            
        return scene_info
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render the environment"""
        if self.env is None:
            return None
        
        return self.env.render(mode=mode)
    
    def start_recording(self, output_path: str, fps: int = DEFAULT_FPS, 
                       resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION):
        """Start video recording"""
        if self._recording:
            self.stop_recording()
        
        self._video_recorder = VideoRecorder(output_path, fps=fps, resolution=resolution)
        self._video_recorder.start_recording()
        self._recording = True
        print(f"Started recording to: {output_path}")
    
    def stop_recording(self) -> Optional[str]:
        """Stop video recording and return path"""
        if self._recording and self._video_recorder:
            self._video_recorder.stop_recording()
            output_path = str(self._video_recorder.output_path)
            self._recording = False
            self._video_recorder = None
            return output_path
        return None
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._recording
    
    def run_interactive(self, window_size: Tuple[int, int] = (800, 600)):
        """Run interactive simulation with pygame controls"""
        try:
            visualizer = PygameVisualizer(window_size=window_size)
            visualizer.run_interactive_simulation(self)
        except ImportError as e:
            print(f"Interactive mode requires pygame: {e}")
            print("Install with: pip install pygame")
    
    def record_episode(self, 
                      strategy: BaseNavigationStrategy,
                      output_path: str,
                      fps: int = DEFAULT_FPS,
                      save_frames: bool = False) -> Dict[str, Any]:
        """Record an episode using a navigation strategy"""
        import time
        
        # Setup recording
        self.start_recording(output_path, fps=fps)
        
        # Reset and run episode
        observations = self.reset()
        step_count = 0
        start_time = time.time()
        
        while True:
            # Add current frame to recording
            if 'rgb' in observations and self._recording:
                self._video_recorder.add_frame(observations['rgb'])
            
            # Check if strategy is done
            if strategy.is_done(observations, step_count):
                break
            
            # Get and execute action
            action = strategy.get_next_action(observations, step_count)
            observations, reward, done, info = self.step(action)
            step_count += 1
            
            # Check if episode is done
            if done:
                break
        
        # Stop recording
        end_time = time.time()
        video_path = self.stop_recording()
        
        # Save frames if requested
        if save_frames and self._video_recorder:
            frames_dir = Path(output_path).parent / f"{Path(output_path).stem}_frames"
            self._video_recorder.save_frames_as_images(str(frames_dir))
        
        return {
            'total_steps': step_count,
            'duration': end_time - start_time,
            'video_path': video_path,
            'fps': fps,
            'episode_stats': self.get_episode_stats()
        }

    def close(self):
        """Close environment and cleanup"""
        # Stop recording if active
        if self._recording:
            self.stop_recording()
        
        if self.env is not None:
            self.env.close()
            self.env = None
    
    @property 
    def observation_space(self):
        """Get observation space"""
        return self.env.observation_space if self.env else None
    
    @property
    def action_space(self):
        """Get action space"""
        return self.env.action_space if self.env else None


class TaskDatasetManager:
    """Manager for different task datasets"""
    
    def __init__(self):
        self.available_datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Discover available datasets"""
        datasets = {}
        
        # Check MP3D
        if MP3D_SCENE_DATASET.exists():
            datasets['mp3d'] = {
                'path': str(MP3D_SCENE_DATASET),
                'tasks': ['pointnav', 'objectnav', 'imagenav'],
                'scenes': self._get_scene_count('mp3d')
            }
        
        # Check HM3D
        if HM3D_SCENE_DATASET.exists():
            datasets['hm3d'] = {
                'path': str(HM3D_SCENE_DATASET),
                'tasks': ['pointnav', 'objectnav'],
                'scenes': self._get_scene_count('hm3d')
            }
        
        return datasets
    
    def _get_scene_count(self, dataset: str) -> int:
        """Get number of scenes in dataset"""
        # This would need to parse the dataset config
        # For now, return a placeholder
        return -1  # Unknown
    
    def list_datasets(self) -> List[str]:
        """List available datasets"""
        return list(self.available_datasets.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset"""
        return self.available_datasets.get(dataset_name)
    
    def create_environment(self, 
                          task_type: str, 
                          dataset_type: str, 
                          **kwargs) -> HabitatEnvironmentWrapper:
        """Create environment for specified task and dataset"""
        
        if dataset_type not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_type} not available. "
                           f"Available: {list(self.available_datasets.keys())}")
        
        config = HabitatWrapperConfig(
            task_type=TaskType(task_type),
            dataset_type=DatasetType(dataset_type),
            **kwargs
        )
        
        return HabitatEnvironmentWrapper(config)


# Navigation strategies for Habitat environments
class HabitatRandomStrategy(BaseNavigationStrategy):
    """Random navigation strategy optimized for Habitat"""
    
    def __init__(self, max_steps: int = 100, forward_prob: float = 0.6):
        self.max_steps = max_steps
        self.forward_prob = forward_prob
        self.actions = ['move_forward', 'turn_left', 'turn_right']
    
    def get_next_action(self, observations: Dict[str, np.ndarray], step_count: int) -> str:
        if np.random.random() < self.forward_prob:
            return 'move_forward'
        else:
            return np.random.choice(['turn_left', 'turn_right'])
    
    def is_done(self, observations: Dict[str, np.ndarray], step_count: int) -> bool:
        return step_count >= self.max_steps


class HabitatPointNavStrategy(BaseNavigationStrategy):
    """Simple point navigation strategy using goal sensor"""
    
    def __init__(self, max_steps: int = 500, success_distance: float = 0.2):
        self.max_steps = max_steps
        self.success_distance = success_distance
    
    def get_next_action(self, observations: Dict[str, np.ndarray], step_count: int) -> str:
        if 'pointgoal' in observations:
            goal = observations['pointgoal']
            distance, angle = goal[0], goal[1]
            
            # Stop if close enough
            if distance < self.success_distance:
                return 'stop'
            
            # Turn towards goal if angle is large
            if abs(angle) > 0.3:  # ~17 degrees
                return 'turn_left' if angle > 0 else 'turn_right'
            else:
                return 'move_forward'
        else:
            # Fallback to random if no goal sensor
            return np.random.choice(['move_forward', 'turn_left', 'turn_right'])
    
    def is_done(self, observations: Dict[str, np.ndarray], step_count: int) -> bool:
        # Stop if reached goal or max steps
        if 'pointgoal' in observations:
            distance = observations['pointgoal'][0]
            if distance < self.success_distance:
                return True
        return step_count >= self.max_steps


class HabitatExplorationStrategy(BaseNavigationStrategy):
    """Exploration strategy that tries to avoid walls and explore"""
    
    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self.stuck_counter = 0
        self.last_position = None
        self.movement_threshold = 0.05
    
    def get_next_action(self, observations: Dict[str, np.ndarray], step_count: int) -> str:
        # Check if we have depth sensor for obstacle avoidance
        if 'depth' in observations:
            depth = observations['depth']
            
            # Check forward obstacle
            center_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
            
            # If obstacle ahead, turn
            if center_depth < 0.5:  # Less than 0.5m ahead
                return 'turn_right' if np.random.random() > 0.5 else 'turn_left'
        
        # Check if stuck (using GPS if available)
        if 'gps' in observations:
            current_pos = observations['gps']
            if self.last_position is not None:
                movement = np.linalg.norm(current_pos - self.last_position)
                if movement < self.movement_threshold:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
            self.last_position = current_pos.copy()
            
            # If stuck, turn
            if self.stuck_counter > 3:
                return 'turn_right'
        
        # Default: move forward with some randomness
        if np.random.random() < 0.8:
            return 'move_forward'
        else:
            return np.random.choice(['turn_left', 'turn_right'])
    
    def is_done(self, observations: Dict[str, np.ndarray], step_count: int) -> bool:
        return step_count >= self.max_steps


# Convenience function for quick environment creation
def create_habitat_env(task: str = "pointnav", 
                      dataset: str = "mp3d", 
                      **kwargs) -> HabitatEnvironmentWrapper:
    """Quick environment creation function"""
    manager = TaskDatasetManager()
    return manager.create_environment(task, dataset, **kwargs)