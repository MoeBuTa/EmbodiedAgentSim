"""
Base task class for embodied AI
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from easim.tasks.base.episode import BaseEpisode, Position, EpisodeDataset, EpisodeIterator
from easim.tasks.base.metrics import BaseMetric, MetricCalculator, MetricResult
from easim.tasks.base.configs import TaskConfig
from easim.core.simulator import CoreSimulator
from easim.core.agents import BaseAgent


class TaskState:
    """Current state of task execution"""

    def __init__(self):
        self.current_episode: Optional[BaseEpisode] = None
        self.episode_step: int = 0
        self.total_reward: float = 0.0
        self.agent_path: List[Position] = []
        self.collision_count: int = 0
        self.is_episode_done: bool = False
        self.episode_success: bool = False
        self.info: Dict[str, Any] = {}

    def reset(self):
        """Reset task state"""
        self.current_episode = None
        self.episode_step = 0
        self.total_reward = 0.0
        self.agent_path = []
        self.collision_count = 0
        self.is_episode_done = False
        self.episode_success = False
        self.info = {}


class BaseTask(ABC):
    """Abstract base class for embodied AI tasks"""

    def __init__(self,
                 config: TaskConfig,
                 simulator: CoreSimulator,
                 episode_dataset: EpisodeDataset,
                 metrics: Optional[List[BaseMetric]] = None):
        self.config = config
        self.simulator = simulator
        self.episode_dataset = episode_dataset
        self.episode_iterator = EpisodeIterator(
            episode_dataset.episodes,
            shuffle=config.shuffle_episodes
        )

        # Initialize metrics
        if metrics is None:
            metrics = self._get_default_metrics()
        self.metric_calculator = MetricCalculator(metrics)

        # Task state
        self.state = TaskState()

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _get_default_metrics(self) -> List[BaseMetric]:
        """Get default metrics for this task"""
        pass

    @abstractmethod
    def _check_success(self,
                      episode: BaseEpisode,
                      agent_position: Position,
                      observations: Dict[str, Any]) -> bool:
        """Check if episode was successful"""
        pass

    @abstractmethod
    def _calculate_reward(self,
                         episode: BaseEpisode,
                         observations: Dict[str, Any],
                         action: str,
                         success: bool) -> float:
        """Calculate reward for current step"""
        pass

    @abstractmethod
    def _get_episode_info(self, episode: BaseEpisode) -> Dict[str, Any]:
        """Get episode-specific information for observations"""
        pass

    def reset(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset task to new episode"""
        # Get episode
        if episode_id is not None:
            episode = self.episode_dataset.get_episode_by_id(episode_id)
            if episode is None:
                raise ValueError(f"Episode {episode_id} not found")
        else:
            try:
                episode = next(self.episode_iterator)
            except StopIteration:
                # Reset iterator and try again
                self.episode_iterator.reset()
                episode = next(self.episode_iterator)

        # Reset task state
        self.state.reset()
        self.state.current_episode = episode

        # Reset simulator agent to episode start
        self.simulator.set_agent_state(
            position=episode.get_start_position(),
            rotation=episode.get_start_rotation()
        )

        # Reset simulator
        self.simulator.reset()

        # Get initial observations
        observations = self.simulator.get_observations()

        # Add task-specific information
        task_info = self._get_episode_info(episode)
        observations.update(task_info)

        # Initialize agent path
        agent_state = self.simulator.get_agent_state()
        start_pos = Position(
            agent_state.position[0],
            agent_state.position[1],
            agent_state.position[2]
        )
        self.state.agent_path = [start_pos]

        self.logger.info(f"Reset to episode {episode.episode_id} in scene {episode.scene_id}")

        return observations

    def step(self, action: Union[str, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the task"""
        if self.state.current_episode is None:
            raise RuntimeError("Task not reset - call reset() first")

        episode = self.state.current_episode

        # Take action in simulator
        observations = self.simulator.step(action)

        # Update step count
        self.state.episode_step += 1

        # Get agent position
        agent_state = self.simulator.get_agent_state()
        agent_position = Position(
            agent_state.position[0],
            agent_state.position[1],
            agent_state.position[2]
        )
        self.state.agent_path.append(agent_position)

        # Check for collisions (if collision sensor available)
        collision = False
        if 'collisions' in observations:
            collision = observations['collisions']['is_collision']
            if collision:
                self.state.collision_count += 1

        # Check success
        success = self._check_success(episode, agent_position, observations)
        self.state.episode_success = success

        # Calculate reward
        reward = self._calculate_reward(episode, observations, action, success)
        self.state.total_reward += reward

        # Check if episode is done
        done = (
            success or
            self.state.episode_step >= self.config.max_episode_steps or
            self._check_early_termination(episode, observations)
        )
        self.state.is_episode_done = done

        # Add task-specific information to observations
        task_info = self._get_episode_info(episode)
        observations.update(task_info)

        # Prepare info dict
        info = {
            "episode_step": self.state.episode_step,
            "episode_id": episode.episode_id,
            "scene_id": episode.scene_id,
            "success": success,
            "collision": collision,
            "total_collisions": self.state.collision_count,
            "distance_to_goal": self._get_distance_to_goal(episode, agent_position),
            "path_length": self._calculate_path_length(self.state.agent_path),
            **self.state.info
        }

        # Update metrics if episode is done
        if done:
            self._update_episode_metrics(episode, success, info)

        return observations, reward, done, info

    def _check_early_termination(self,
                                episode: BaseEpisode,
                                observations: Dict[str, Any]) -> bool:
        """Check for early termination conditions"""
        # Override in subclasses for task-specific termination
        return False

    def _get_distance_to_goal(self,
                             episode: BaseEpisode,
                             agent_position: Position) -> float:
        """Calculate distance to goal"""
        if not episode.goals or not episode.goals[0].position:
            return float('inf')

        goal_pos = episode.goals[0].position.to_array()
        agent_pos = agent_position.to_array()
        return float(np.linalg.norm(agent_pos - goal_pos))

    def _calculate_path_length(self, path: List[Position]) -> float:
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path)):
            pos1 = path[i-1].to_array()
            pos2 = path[i].to_array()
            total_length += np.linalg.norm(pos2 - pos1)

        return total_length

    def _update_episode_metrics(self,
                               episode: BaseEpisode,
                               success: bool,
                               info: Dict[str, Any]):
        """Update metrics with completed episode"""
        self.metric_calculator.update_episode(
            episode=episode,
            agent_path=self.state.agent_path,
            success=success,
            num_steps=self.state.episode_step,
            num_collisions=self.state.collision_count,
            final_distance=info.get("distance_to_goal"),
            **info
        )

    def get_current_episode(self) -> Optional[BaseEpisode]:
        """Get current episode"""
        return self.state.current_episode

    def get_episode_metrics(self) -> Dict[str, MetricResult]:
        """Get current episode metrics"""
        return self.metric_calculator.compute_final_metrics()

    def reset_metrics(self):
        """Reset all metrics"""
        self.metric_calculator.reset()

    def get_metric_summary(self) -> Dict[str, float]:
        """Get summary of metric values"""
        return self.metric_calculator.get_metric_summary()

    def is_done(self) -> bool:
        """Check if current episode is done"""
        return self.state.is_episode_done

    def get_task_info(self) -> Dict[str, Any]:
        """Get general task information"""
        return {
            "task_type": self.__class__.__name__,
            "max_episode_steps": self.config.max_episode_steps,
            "success_distance": self.config.success_distance,
            "num_episodes": len(self.episode_dataset),
            "current_episode": self.state.current_episode.episode_id if self.state.current_episode else None
        }


class NavigationTask(BaseTask):
    """Base class for navigation tasks"""

    def _get_default_metrics(self) -> List[BaseMetric]:
        """Get default navigation metrics"""
        from .metrics import get_navigation_metrics
        return get_navigation_metrics()

    def _check_success(self,
                      episode: BaseEpisode,
                      agent_position: Position,
                      observations: Dict[str, Any]) -> bool:
        """Check navigation success"""
        if not episode.goals or not episode.goals[0].position:
            return False

        distance = self._get_distance_to_goal(episode, agent_position)
        success_distance = episode.goals[0].radius or self.config.success_distance

        return distance <= success_distance

    def _calculate_reward(self,
                         episode: BaseEpisode,
                         observations: Dict[str, Any],
                         action: str,
                         success: bool) -> float:
        """Calculate navigation reward"""
        reward = self.config.step_penalty  # Small penalty for each step

        if success:
            reward += self.config.success_reward

        # Collision penalty
        if observations.get('collisions', {}).get('is_collision', False):
            reward += self.config.collision_penalty

        # Distance-based shaping (optional)
        if hasattr(self, '_previous_distance_to_goal'):
            current_distance = self._get_distance_to_goal(
                episode,
                Position(
                    observations['agent_position'][0],
                    observations['agent_position'][1],
                    observations['agent_position'][2]
                )
            )
            distance_reward = self._previous_distance_to_goal - current_distance
            reward += distance_reward * 0.1  # Small shaping reward
            self._previous_distance_to_goal = current_distance

        return reward

    def _get_episode_info(self, episode: BaseEpisode) -> Dict[str, Any]:
        """Get navigation episode info"""
        info = {}

        # Add goal information
        if episode.goals:
            goal = episode.goals[0]
            if goal.position:
                info['goal_position'] = goal.position.to_list()
            if goal.object_category:
                info['target_object_category'] = goal.object_category

        return info


class PointNavigationTask(NavigationTask):
    """Point Navigation task"""

    def _get_episode_info(self, episode: BaseEpisode) -> Dict[str, Any]:
        """Get PointNav episode info"""
        info = super()._get_episode_info(episode)

        # Add pointgoal sensor information
        if episode.goals and episode.goals[0].position:
            # This would be handled by the pointgoal sensor in habitat
            info['pointgoal_with_gps_compass'] = episode.goals[0].position.to_list()

        return info


class ObjectNavigationTask(NavigationTask):
    """Object Navigation task"""

    def _get_episode_info(self, episode: BaseEpisode) -> Dict[str, Any]:
        """Get ObjectNav episode info"""
        info = super()._get_episode_info(episode)

        # Add object goal information
        if episode.goals and episode.goals[0].object_category:
            info['objectgoal'] = episode.goals[0].object_category

        return info


class TaskRunner:
    """Runs tasks and collects results"""

    def __init__(self, task: BaseTask):
        self.task = task
        self.episode_results = []

    def run_episode(self,
                   episode_id: Optional[str] = None,
                   max_steps: Optional[int] = None,
                   action_policy: Optional[callable] = None) -> Dict[str, Any]:
        """Run a single episode"""
        # Reset task
        observations = self.task.reset(episode_id)
        episode = self.task.get_current_episode()

        if max_steps is None:
            max_steps = self.task.config.max_episode_steps

        # Episode loop
        step_count = 0
        trajectory = []

        while not self.task.is_done() and step_count < max_steps:
            # Get action
            if action_policy:
                action = action_policy(observations, step_count)
            else:
                # Random action
                agent = self.task.simulator.get_agent()
                action_names = agent.get_action_names() if agent else ["move_forward"]
                action = np.random.choice(action_names)

            # Take step
            observations, reward, done, info = self.task.step(action)

            # Record trajectory
            trajectory.append({
                "step": step_count,
                "action": action,
                "reward": reward,
                "observations": {k: v for k, v in observations.items()
                               if k not in ['rgb', 'depth', 'semantic']},  # Exclude images
                "info": info
            })

            step_count += 1

        # Compile results
        final_info = trajectory[-1]['info'] if trajectory else {}
        result = {
            "episode_id": episode.episode_id if episode else "unknown",
            "scene_id": episode.scene_id if episode else "unknown",
            "success": final_info.get('success', False),
            "steps": step_count,
            "path_length": final_info.get('path_length', 0.0),
            "distance_to_goal": final_info.get('distance_to_goal', float('inf')),
            "collisions": final_info.get('total_collisions', 0),
            "trajectory": trajectory
        }

        self.episode_results.append(result)
        return result

    def run_evaluation(self,
                      num_episodes: Optional[int] = None,
                      action_policy: Optional[callable] = None) -> Dict[str, Any]:
        """Run evaluation on multiple episodes"""
        if num_episodes is None:
            num_episodes = len(self.task.episode_dataset)

        self.task.reset_metrics()
        results = []

        for i in range(num_episodes):
            try:
                result = self.run_episode(action_policy=action_policy)
                results.append(result)

                if (i + 1) % 10 == 0:
                    self.task.logger.info(f"Completed {i + 1}/{num_episodes} episodes")

            except StopIteration:
                self.task.logger.info(f"Ran out of episodes after {i} episodes")
                break
            except Exception as e:
                self.task.logger.error(f"Error in episode {i}: {e}")
                continue

        # Get final metrics
        metrics = self.task.get_metric_summary()

        # Compile evaluation results
        evaluation_result = {
            "num_episodes": len(results),
            "metrics": metrics,
            "episode_results": results,
            "task_info": self.task.get_task_info()
        }

        return evaluation_result

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.episode_results:
            return {}

        successes = [r['success'] for r in self.episode_results]
        steps = [r['steps'] for r in self.episode_results]
        path_lengths = [r['path_length'] for r in self.episode_results]
        distances = [r['distance_to_goal'] for r in self.episode_results]

        return {
            "num_episodes": len(self.episode_results),
            "success_rate": np.mean(successes),
            "avg_steps": np.mean(steps),
            "avg_path_length": np.mean(path_lengths),
            "avg_distance_to_goal": np.mean([d for d in distances if d != float('inf')]),
            "total_collisions": sum(r['collisions'] for r in self.episode_results)
        }