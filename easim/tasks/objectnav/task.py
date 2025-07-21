"""
Object Navigation task implementation
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from easim.tasks.base.task import NavigationTask
from easim.tasks.base.episode import ObjectNavEpisode, Position
from easim.tasks.base.metrics import BaseMetric, get_objectnav_metrics
from easim.tasks.base.configs import TaskConfig
from easim.tasks.objectnav.configs import ObjectNavConfig
from easim.core.simulator import CoreSimulator
from easim.datasets.base.dataset import BaseDataset


class ObjectNavigationTask(NavigationTask):
    """Object Navigation task"""

    def __init__(self,
                 config: ObjectNavConfig,
                 simulator: CoreSimulator,
                 episode_dataset: 'ObjectNavDataset',
                 metrics: Optional[List[BaseMetric]] = None):

        # Ensure we have ObjectNav episodes
        if episode_dataset.episodes and not isinstance(episode_dataset.episodes[0], ObjectNavEpisode):
            raise ValueError("ObjectNavTask requires ObjectNavEpisode instances")

        super().__init__(config, simulator, episode_dataset, metrics)
        self.objectnav_config = config

    def _get_default_metrics(self) -> List[BaseMetric]:
        """Get ObjectNav specific metrics"""
        return get_objectnav_metrics()

    def _check_success(self,
                      episode: ObjectNavEpisode,
                      agent_position: Position,
                      observations: Dict[str, Any]) -> bool:
        """Check ObjectNav success"""
        if not episode.goals:
            return False

        target_category = episode.goals[0].object_category
        if not target_category:
            return False

        if self.objectnav_config.view_success:
            return self._check_view_success(episode, observations)
        else:
            return self._check_distance_success(episode, agent_position)

    def _check_view_success(self,
                           episode: ObjectNavEpisode,
                           observations: Dict[str, Any]) -> bool:
        """Check if target object is in view"""
        target_category = episode.goals[0].object_category

        # Check semantic sensor for target object
        if 'semantic' in observations:
            semantic_obs = observations['semantic']

            # This would require scene-specific object mappings
            # For now, implement basic distance-based success
            return self._check_distance_success(episode, self._get_agent_position(observations))

        return False

    def _check_distance_success(self,
                               episode: ObjectNavEpisode,
                               agent_position: Position) -> bool:
        """Check distance-based success"""
        if not episode.goals[0].position:
            return False

        distance = self._get_distance_to_goal(episode, agent_position)
        success_distance = episode.goals[0].radius or self.config.success_distance

        return distance <= success_distance

    def _get_agent_position(self, observations: Dict[str, Any]) -> Position:
        """Extract agent position from observations"""
        # This would typically come from GPS sensor
        if 'gps' in observations:
            pos = observations['gps']
            return Position(pos[0], 0.0, pos[1])  # GPS is typically 2D

        # Fallback to simulator state
        agent_state = self.simulator.get_agent_state()
        return Position(
            agent_state.position[0],
            agent_state.position[1],
            agent_state.position[2]
        )

    def _calculate_reward(self,
                         episode: ObjectNavEpisode,
                         observations: Dict[str, Any],
                         action: str,
                         success: bool) -> float:
        """Calculate ObjectNav reward"""
        reward = super()._calculate_reward(episode, observations, action, success)

        # Additional ObjectNav-specific rewards
        target_category = episode.goals[0].object_category

        # View reward bonus
        if self._object_in_view(observations, target_category):
            reward += self.objectnav_config.view_reward_bonus

        # Category reward for seeing target category
        if self._target_category_visible(observations, target_category):
            reward += self.objectnav_config.category_reward

        return reward

    def _object_in_view(self, observations: Dict[str, Any], target_category: str) -> bool:
        """Check if target object is visible"""
        # This would require semantic segmentation analysis
        # Placeholder implementation
        return False

    def _target_category_visible(self, observations: Dict[str, Any], target_category: str) -> bool:
        """Check if target category is visible in semantic sensor"""
        # This would require scene-specific object category mappings
        # Placeholder implementation
        return False

    def _get_episode_info(self, episode: ObjectNavEpisode) -> Dict[str, Any]:
        """Get ObjectNav episode info"""
        info = super()._get_episode_info(episode)

        # Add ObjectNav specific information
        if episode.goals and episode.goals[0].object_category:
            target_category = episode.goals[0].object_category
            info.update({
                'objectgoal': target_category,
                'objectgoal_id': self.objectnav_config.object_category_mapping.get(target_category, -1)
            })

        return info

    def get_target_categories(self) -> List[str]:
        """Get list of target object categories"""
        return self.objectnav_config.target_object_categories

    def get_category_id(self, category: str) -> int:
        """Get category ID for object category"""
        return self.objectnav_config.object_category_mapping.get(category, -1)

    def get_category_name(self, category_id: int) -> str:
        """Get category name for category ID"""
        for name, cid in self.objectnav_config.object_category_mapping.items():
            if cid == category_id:
                return name
        return "unknown"


# Factory function for creating ObjectNav tasks
def create_objectnav_task(dataset_name: str = "hm3d",
                         split: str = "val",
                         max_episodes: Optional[int] = None,
                         **kwargs) -> ObjectNavigationTask:
    """Create ObjectNav task"""
    from easim.core.simulator import TaskSimulator
    from easim.datasets.hm3d.objectnav import HM3DObjectNavDataset
    from easim.datasets.mp3d.objectnav import MP3DObjectNavDataset

    # Create simulator
    simulator = TaskSimulator(
        task_type="objectnav",
        dataset_type=dataset_name.upper(),
        **kwargs
    )

    # Load dataset
    if dataset_name.lower() == "hm3d":
        dataset = HM3DObjectNavDataset(split=split, max_episodes=max_episodes)
    elif dataset_name.lower() == "mp3d":
        dataset = MP3DObjectNavDataset(split=split, max_episodes=max_episodes)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create config
    config = ObjectNavConfig(
        split=split,
        max_episodes=max_episodes,
        **kwargs
    )

    return ObjectNavigationTask(config, simulator, dataset)


# Evaluation utilities
def evaluate_objectnav_agent(agent_policy: callable,
                            dataset_name: str = "hm3d",
                            split: str = "val",
                            num_episodes: int = 100,
                            **kwargs) -> Dict[str, Any]:
    """Evaluate ObjectNav agent"""
    from easim.tasks.base.task import TaskRunner

    # Create task
    task = create_objectnav_task(
        dataset_name=dataset_name,
        split=split,
        max_episodes=num_episodes,
        **kwargs
    )

    # Run evaluation
    runner = TaskRunner(task)
    results = runner.run_evaluation(
        num_episodes=num_episodes,
        action_policy=agent_policy
    )

    return results