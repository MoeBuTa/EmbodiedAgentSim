import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from easim.utils.config import hm3d_config, mp3d_config, r2r_config

# Import habitat only when needed to avoid dependency issues
try:
    import habitat

    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: habitat module not available, task environments disabled")


@dataclass
class TaskEpisode:
    """Represents a single task episode"""
    episode_id: str
    scene_id: str
    start_position: List[float]
    start_rotation: List[float]
    goals: List[Dict[str, Any]]
    instruction: Optional[str] = None
    shortest_paths: Optional[List[List[Dict]]] = None


class BaseTaskEnvironment(ABC):
    """Abstract base class for task environments"""

    def __init__(self, split: str = "val", max_episodes: int = 100):
        if not HABITAT_AVAILABLE:
            raise ImportError("habitat module is required for task environments")

        self.split = split
        self.max_episodes = max_episodes
        self.env = None
        self.current_episode = None
        self._initialize_environment()

    def _initialize_environment(self):
        """Initialize the habitat environment"""
        config = self._get_task_config()
        if config is not None:
            self.env = habitat.Env(config=config)
        else:
            raise RuntimeError("Could not create task configuration")

    @abstractmethod
    def _get_task_config(self):
        """Get task-specific configuration"""
        pass

    def reset(self) -> Dict[str, Any]:
        """Reset environment to start of episode"""
        if self.env is None:
            raise RuntimeError("Environment not initialized")

        observations = self.env.reset()
        self.current_episode = self._parse_current_episode()
        return observations

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take action and return observations, reward, done, info"""
        if self.env is None:
            raise RuntimeError("Environment not initialized")

        observations = self.env.step(action)

        # Get metrics and info
        metrics = self.env.get_metrics()
        done = self.env.episode_over

        # Calculate reward (can be customized per task)
        reward = self._calculate_reward(observations, metrics, done)

        return observations, reward, done, metrics

    def _calculate_reward(self, observations: Dict[str, Any],
                          metrics: Dict[str, Any], done: bool) -> float:
        """Calculate reward based on task progress"""
        # Default sparse reward
        if done and metrics.get("success", False):
            return 1.0
        return 0.0

    def _parse_current_episode(self) -> TaskEpisode:
        """Parse current episode information"""
        episode = self.env.current_episode

        return TaskEpisode(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            start_position=episode.start_position.tolist(),
            start_rotation=episode.start_rotation.tolist(),
            goals=[goal.__dict__ for goal in episode.goals],
            instruction=getattr(episode, 'instruction', None),
            shortest_paths=getattr(episode, 'shortest_paths', None)
        )

    def get_current_episode_info(self) -> Optional[TaskEpisode]:
        """Get information about current episode"""
        return self.current_episode

    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()


class HM3DObjectNavEnvironment(BaseTaskEnvironment):
    """Environment for HM3D ObjectNav task"""

    def __init__(self, split: str = "val", max_episodes: int = 100):
        super().__init__(split, max_episodes)

    def _get_task_config(self):
        """Get HM3D ObjectNav configuration"""
        return hm3d_config(stage=self.split, episodes=self.max_episodes)

    def get_target_object_category(self) -> Optional[str]:
        """Get target object category for current episode"""
        if self.current_episode and self.current_episode.goals:
            return self.current_episode.goals[0].get('object_category')
        return None

    def _calculate_reward(self, observations: Dict[str, Any],
                          metrics: Dict[str, Any], done: bool) -> float:
        """ObjectNav specific reward calculation"""
        if done:
            if metrics.get("success", False):
                return 10.0  # Success reward
            else:
                return -1.0  # Failure penalty

        # Distance-based reward shaping
        distance_to_goal = metrics.get("distance_to_goal", float('inf'))
        if hasattr(self, '_previous_distance'):
            distance_reward = self._previous_distance - distance_to_goal
            self._previous_distance = distance_to_goal
            return distance_reward * 0.1  # Small reward for getting closer

        self._previous_distance = distance_to_goal
        return 0.0


class MP3DObjectNavEnvironment(BaseTaskEnvironment):
    """Environment for MP3D ObjectNav task"""

    def __init__(self, split: str = "val", max_episodes: int = 100):
        super().__init__(split, max_episodes)

    def _get_task_config(self):
        """Get MP3D ObjectNav configuration"""
        return mp3d_config(stage=self.split, episodes=self.max_episodes)

    def get_target_object_category(self) -> Optional[str]:
        """Get target object category for current episode"""
        if self.current_episode and self.current_episode.goals:
            return self.current_episode.goals[0].get('object_category')
        return None


class R2RNavigationEnvironment(BaseTaskEnvironment):
    """Environment for Room-to-Room (R2R) VLN task"""

    def __init__(self, split: str = "val_seen", max_episodes: int = 100):
        super().__init__(split, max_episodes)

    def _get_task_config(self):
        """Get R2R VLN configuration"""
        return r2r_config(stage=self.split, episodes=self.max_episodes)

    def get_instruction(self) -> Optional[str]:
        """Get navigation instruction for current episode"""
        if self.current_episode:
            return self.current_episode.instruction
        return None

    def _calculate_reward(self, observations: Dict[str, Any],
                          metrics: Dict[str, Any], done: bool) -> float:
        """R2R VLN specific reward calculation"""
        if done:
            success = metrics.get("success", False)
            spl = metrics.get("spl", 0.0)  # Success weighted by Path Length

            if success:
                return 10.0 + spl * 5.0  # Bonus for efficient paths
            else:
                return -1.0

        # Progress reward based on distance to goal
        distance_to_goal = metrics.get("distance_to_goal", float('inf'))
        if hasattr(self, '_previous_distance'):
            progress = self._previous_distance - distance_to_goal
            self._previous_distance = distance_to_goal
            return progress * 0.1

        self._previous_distance = distance_to_goal
        return 0.0


class TaskEnvironmentFactory:
    """Factory for creating task environments"""

    @staticmethod
    def create_environment(task_type: str, split: str = "val",
                           max_episodes: int = 100) -> BaseTaskEnvironment:
        """Create task environment based on type"""
        if not HABITAT_AVAILABLE:
            raise ImportError("habitat module is required for task environments")

        task_type = task_type.upper()

        if task_type == "HM3D_OBJECTNAV":
            return HM3DObjectNavEnvironment(split, max_episodes)
        elif task_type == "MP3D_OBJECTNAV":
            return MP3DObjectNavEnvironment(split, max_episodes)
        elif task_type == "R2R_VLN":
            return R2RNavigationEnvironment(split, max_episodes)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


class TaskRunner:
    """Runs tasks and collects results"""

    def __init__(self, environment: BaseTaskEnvironment):
        self.environment = environment
        self.episode_results = []

    def run_episode(self, max_steps: int = 500,
                    action_strategy: Optional[callable] = None) -> Dict[str, Any]:
        """Run a single episode"""
        observations = self.environment.reset()
        episode_info = self.environment.get_current_episode_info()

        step_count = 0
        total_reward = 0.0
        done = False
        trajectory = []

        while not done and step_count < max_steps:
            # Get action (random if no strategy provided)
            if action_strategy:
                action = action_strategy(observations, step_count)
            else:
                action = np.random.randint(0, 4)  # Random action

            # Take step
            observations, reward, done, metrics = self.environment.step(action)

            # Record trajectory
            trajectory.append({
                "step": step_count,
                "action": action,
                "reward": reward,
                "position": observations.get("agent_position", []),
                "rotation": observations.get("agent_rotation", [])
            })

            total_reward += reward
            step_count += 1

        # Compile episode results
        result = {
            "episode_id": episode_info.episode_id if episode_info else "unknown",
            "scene_id": episode_info.scene_id if episode_info else "unknown",
            "instruction": episode_info.instruction if episode_info else None,
            "target_object": getattr(self.environment, 'get_target_object_category', lambda: None)(),
            "steps": step_count,
            "total_reward": total_reward,
            "success": metrics.get("success", False),
            "spl": metrics.get("spl", 0.0),
            "distance_to_goal": metrics.get("distance_to_goal", float('inf')),
            "trajectory": trajectory,
            "final_metrics": metrics
        }

        self.episode_results.append(result)
        return result

    def run_multiple_episodes(self, num_episodes: int, max_steps: int = 500,
                              action_strategy: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Run multiple episodes and return results"""
        results = []

        for i in range(num_episodes):
            print(f"Running episode {i + 1}/{num_episodes}")
            result = self.run_episode(max_steps, action_strategy)
            results.append(result)

            # Print episode summary
            print(f"Episode {i + 1}: Success={result['success']}, "
                  f"Steps={result['steps']}, Reward={result['total_reward']:.2f}")

        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all episodes"""
        if not self.episode_results:
            return {}

        successes = [r['success'] for r in self.episode_results]
        steps = [r['steps'] for r in self.episode_results]
        rewards = [r['total_reward'] for r in self.episode_results]
        spls = [r['spl'] for r in self.episode_results]

        return {
            "num_episodes": len(self.episode_results),
            "success_rate": np.mean(successes),
            "average_steps": np.mean(steps),
            "average_reward": np.mean(rewards),
            "average_spl": np.mean(spls),
            "std_steps": np.std(steps),
            "std_reward": np.std(rewards)
        }