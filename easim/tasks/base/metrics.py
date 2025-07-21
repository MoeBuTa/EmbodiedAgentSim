"""
Base metrics for embodied AI tasks
"""
import numpy as np
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from easim.tasks.base.episode import BaseEpisode, Position


@dataclass
class MetricResult:
    """Result of a metric calculation"""
    name: str
    value: float
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


class BaseMetric(ABC):
    """Abstract base class for metrics"""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset metric state"""
        pass

    @abstractmethod
    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               **kwargs) -> Optional[MetricResult]:
        """Update metric with episode results"""
        pass

    @abstractmethod
    def compute(self) -> MetricResult:
        """Compute final metric value"""
        pass


class SuccessRate(BaseMetric):
    """Success rate metric"""

    def __init__(self):
        super().__init__("success_rate")

    def reset(self):
        """Reset success rate"""
        self.successes = 0
        self.total_episodes = 0

    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               **kwargs) -> Optional[MetricResult]:
        """Update success rate"""
        self.total_episodes += 1
        if success:
            self.successes += 1

        return MetricResult(
            name=self.name,
            value=self.successes / self.total_episodes if self.total_episodes > 0 else 0.0,
            info={"successes": self.successes, "total": self.total_episodes}
        )

    def compute(self) -> MetricResult:
        """Compute final success rate"""
        return MetricResult(
            name=self.name,
            value=self.successes / self.total_episodes if self.total_episodes > 0 else 0.0,
            info={"successes": self.successes, "total": self.total_episodes}
        )


class PathLength(BaseMetric):
    """Path length metric"""

    def __init__(self):
        super().__init__("path_length")

    def reset(self):
        """Reset path length"""
        self.path_lengths = []

    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               **kwargs) -> Optional[MetricResult]:
        """Update path length"""
        if len(agent_path) < 2:
            path_length = 0.0
        else:
            path_length = 0.0
            for i in range(1, len(agent_path)):
                pos1 = agent_path[i - 1].to_array()
                pos2 = agent_path[i].to_array()
                path_length += np.linalg.norm(pos2 - pos1)

        self.path_lengths.append(path_length)

        return MetricResult(
            name=self.name,
            value=path_length,
            info={"episode_path_length": path_length}
        )

    def compute(self) -> MetricResult:
        """Compute average path length"""
        avg_length = np.mean(self.path_lengths) if self.path_lengths else 0.0
        return MetricResult(
            name=self.name,
            value=avg_length,
            info={
                "avg_path_length": avg_length,
                "min_path_length": np.min(self.path_lengths) if self.path_lengths else 0.0,
                "max_path_length": np.max(self.path_lengths) if self.path_lengths else 0.0,
                "num_episodes": len(self.path_lengths)
            }
        )


class SPL(BaseMetric):
    """Success weighted by Path Length (SPL)"""

    def __init__(self):
        super().__init__("spl")

    def reset(self):
        """Reset SPL"""
        self.spl_values = []

    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               optimal_path_length: Optional[float] = None,
               **kwargs) -> Optional[MetricResult]:
        """Update SPL"""
        if not success:
            spl_value = 0.0
        else:
            # Calculate agent path length
            if len(agent_path) < 2:
                agent_path_length = 0.0
            else:
                agent_path_length = 0.0
                for i in range(1, len(agent_path)):
                    pos1 = agent_path[i - 1].to_array()
                    pos2 = agent_path[i].to_array()
                    agent_path_length += np.linalg.norm(pos2 - pos1)

            # Use provided optimal path length or estimate
            if optimal_path_length is None:
                if hasattr(episode, 'get_path_length'):
                    optimal_path_length = episode.get_path_length()
                else:
                    optimal_path_length = agent_path_length  # Fallback

            if optimal_path_length > 0:
                spl_value = optimal_path_length / max(agent_path_length, optimal_path_length)
            else:
                spl_value = 1.0 if success else 0.0

        self.spl_values.append(spl_value)

        return MetricResult(
            name=self.name,
            value=spl_value,
            info={"episode_spl": spl_value}
        )

    def compute(self) -> MetricResult:
        """Compute average SPL"""
        avg_spl = np.mean(self.spl_values) if self.spl_values else 0.0
        return MetricResult(
            name=self.name,
            value=avg_spl,
            info={
                "avg_spl": avg_spl,
                "num_episodes": len(self.spl_values)
            }
        )


class DistanceToGoal(BaseMetric):
    """Distance to goal at episode end"""

    def __init__(self):
        super().__init__("distance_to_goal")

    def reset(self):
        """Reset distance to goal"""
        self.distances = []

    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               final_distance: Optional[float] = None,
               **kwargs) -> Optional[MetricResult]:
        """Update distance to goal"""
        if final_distance is None:
            # Calculate distance from final position to goal
            if agent_path and episode.goals:
                final_pos = agent_path[-1].to_array()
                goal_pos = episode.goals[0].position.to_array()
                final_distance = np.linalg.norm(final_pos - goal_pos)
            else:
                final_distance = float('inf')

        self.distances.append(final_distance)

        return MetricResult(
            name=self.name,
            value=final_distance,
            info={"episode_distance": final_distance}
        )

    def compute(self) -> MetricResult:
        """Compute average distance to goal"""
        avg_distance = np.mean(self.distances) if self.distances else float('inf')
        return MetricResult(
            name=self.name,
            value=avg_distance,
            info={
                "avg_distance": avg_distance,
                "min_distance": np.min(self.distances) if self.distances else float('inf'),
                "max_distance": np.max(self.distances) if self.distances else 0.0,
                "num_episodes": len(self.distances)
            }
        )


class Steps(BaseMetric):
    """Number of steps taken"""

    def __init__(self):
        super().__init__("steps")

    def reset(self):
        """Reset steps"""
        self.step_counts = []

    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               num_steps: Optional[int] = None,
               **kwargs) -> Optional[MetricResult]:
        """Update steps"""
        if num_steps is None:
            num_steps = len(agent_path) - 1 if len(agent_path) > 0 else 0

        self.step_counts.append(num_steps)

        return MetricResult(
            name=self.name,
            value=num_steps,
            info={"episode_steps": num_steps}
        )

    def compute(self) -> MetricResult:
        """Compute average steps"""
        avg_steps = np.mean(self.step_counts) if self.step_counts else 0.0
        return MetricResult(
            name=self.name,
            value=avg_steps,
            info={
                "avg_steps": avg_steps,
                "min_steps": np.min(self.step_counts) if self.step_counts else 0,
                "max_steps": np.max(self.step_counts) if self.step_counts else 0,
                "num_episodes": len(self.step_counts)
            }
        )


class CollisionCount(BaseMetric):
    """Number of collisions"""

    def __init__(self):
        super().__init__("collisions")

    def reset(self):
        """Reset collisions"""
        self.collision_counts = []

    def update(self,
               episode: BaseEpisode,
               agent_path: List[Position],
               success: bool,
               num_collisions: int = 0,
               **kwargs) -> Optional[MetricResult]:
        """Update collisions"""
        self.collision_counts.append(num_collisions)

        return MetricResult(
            name=self.name,
            value=num_collisions,
            info={"episode_collisions": num_collisions}
        )

    def compute(self) -> MetricResult:
        """Compute average collisions"""
        avg_collisions = np.mean(self.collision_counts) if self.collision_counts else 0.0
        return MetricResult(
            name=self.name,
            value=avg_collisions,
            info={
                "avg_collisions": avg_collisions,
                "total_collisions": sum(self.collision_counts),
                "num_episodes": len(self.collision_counts)
            }
        )


class MetricCalculator:
    """Calculates multiple metrics for episodes"""

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = {metric.name: metric for metric in metrics}

    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()

    def update_episode(self,
                       episode: BaseEpisode,
                       agent_path: List[Position],
                       success: bool,
                       **kwargs) -> Dict[str, MetricResult]:
        """Update all metrics with episode results"""
        results = {}
        for name, metric in self.metrics.items():
            result = metric.update(episode, agent_path, success, **kwargs)
            if result:
                results[name] = result
        return results

    def compute_final_metrics(self) -> Dict[str, MetricResult]:
        """Compute final metric values"""
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        return results

    def get_metric_summary(self) -> Dict[str, float]:
        """Get summary of metric values"""
        final_metrics = self.compute_final_metrics()
        return {name: result.value for name, result in final_metrics.items()}


# Standard metric sets for different tasks
def get_navigation_metrics() -> List[BaseMetric]:
    """Get standard navigation metrics"""
    return [
        SuccessRate(),
        SPL(),
        PathLength(),
        DistanceToGoal(),
        Steps()
    ]


def get_objectnav_metrics() -> List[BaseMetric]:
    """Get ObjectNav specific metrics"""
    return get_navigation_metrics()


def get_pointnav_metrics() -> List[BaseMetric]:
    """Get PointNav specific metrics"""
    return get_navigation_metrics()


def create_metric_calculator(task_type: str) -> MetricCalculator:
    """Create metric calculator for task type"""
    task_type = task_type.lower()

    if task_type in ["navigation", "nav"]:
        metrics = get_navigation_metrics()
    elif task_type in ["pointnav", "point_navigation"]:
        metrics = get_pointnav_metrics()
    elif task_type in ["objectnav", "object_navigation"]:
        metrics = get_objectnav_metrics()
    else:
        # Default to navigation metrics
        metrics = get_navigation_metrics()

    return MetricCalculator(metrics)