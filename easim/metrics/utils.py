"""Utility functions for metric collection and analysis"""
from typing import Dict, List, Any, Optional, Type, Union
import numpy as np
from pathlib import Path
import json

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

from habitat.core.embodied_task import Measure, Measurements, Metrics


def create_measure_from_config(measure_type: Type[Measure], config: Dict[str, Any], 
                             sim=None, task=None) -> Measure:
    """
    Create a measure instance from configuration
    
    Args:
        measure_type: Class of measure to create
        config: Configuration dictionary
        sim: Simulator instance (if needed)
        task: Task instance (if needed)
        
    Returns:
        Configured measure instance
    """
    return measure_type(sim=sim, config=config, task=task)


def collect_episode_metrics(task, measures: Optional[List[Measure]] = None) -> Dict[str, Any]:
    """
    Collect all metrics from a completed episode
    
    Args:
        task: Task instance that ran the episode
        measures: Optional list of custom measures to include
        
    Returns:
        Dictionary of collected metrics
    """
    metrics = {}
    
    # Get basic task metrics
    if hasattr(task, 'get_episode_stats'):
        metrics.update(task.get_episode_stats())
    
    # Get task info
    if hasattr(task, 'get_task_info'):
        metrics.update(task.get_task_info())
    
    # Get custom measures if provided
    if measures:
        measure_dict = {measure.uuid: measure for measure in measures}
        habitat_metrics = Metrics(measure_dict)
        metrics.update(dict(habitat_metrics))
    
    # Get environment metrics if available
    if hasattr(task, 'env') and hasattr(task.env, 'get_metrics'):
        env_metrics = task.env.get_metrics()
        metrics.update(env_metrics)
    
    return metrics


def analyze_metrics_batch(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a batch of episode metrics to compute statistics
    
    Args:
        metrics_list: List of metric dictionaries from multiple episodes
        
    Returns:
        Dictionary with statistical analysis
    """
    if not metrics_list:
        return {}
    
    analysis = {
        'num_episodes': len(metrics_list),
        'metrics': {}
    }
    
    # Find all numeric metrics
    numeric_metrics = set()
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                numeric_metrics.add(key)
    
    # Calculate statistics for each numeric metric
    for metric_name in numeric_metrics:
        values = []
        for metrics in metrics_list:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    values.append(float(value))
        
        if values:
            analysis['metrics'][metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
    
    # Calculate success rate if success metric exists
    if 'success' in numeric_metrics:
        success_values = [m.get('success', False) for m in metrics_list]
        success_rate = sum(success_values) / len(success_values)
        analysis['success_rate'] = success_rate
    
    # Calculate completion rate (episodes that finished vs timed out)
    steps_values = []
    max_steps_values = []
    for metrics in metrics_list:
        if 'steps' in metrics and 'max_steps' in metrics:
            steps_values.append(metrics['steps'])
            max_steps_values.append(metrics['max_steps'])
    
    if steps_values and max_steps_values:
        completed = sum(1 for s, m in zip(steps_values, max_steps_values) if s < m)
        completion_rate = completed / len(steps_values)
        analysis['completion_rate'] = completion_rate
    
    return analysis


class MetricsCollector:
    """Utility class for collecting and managing metrics during training/evaluation"""
    
    def __init__(self, measures: Optional[List[Measure]] = None):
        """
        Initialize metrics collector
        
        Args:
            measures: List of measures to track
        """
        self.measures = measures or []
        self.episode_metrics = []
        self.current_episode_measures = {}
    
    def reset_episode(self, episode=None):
        """Reset measures for a new episode"""
        self.current_episode_measures = {}
        for measure in self.measures:
            measure.reset_metric(episode=episode)
            self.current_episode_measures[measure.uuid] = measure
    
    def update_step(self, episode=None, action=None, observations=None, task=None):
        """Update measures after each step"""
        for measure in self.measures:
            measure.update_metric(
                episode=episode,
                action=action, 
                observations=observations,
                task=task
            )
    
    def collect_episode(self, task=None) -> Dict[str, Any]:
        """Collect metrics for completed episode"""
        episode_metrics = collect_episode_metrics(task, self.measures)
        self.episode_metrics.append(episode_metrics)
        return episode_metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current measure values"""
        return {uuid: measure.get_metric() 
                for uuid, measure in self.current_episode_measures.items()}
    
    def analyze_all_episodes(self) -> Dict[str, Any]:
        """Analyze all collected episodes"""
        return analyze_metrics_batch(self.episode_metrics)
    
    def save_metrics(self, filepath: Union[str, Path]):
        """Save collected metrics to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'episode_metrics': self.episode_metrics,
            'analysis': self.analyze_all_episodes()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_metrics(self, filepath: Union[str, Path]):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episode_metrics = data.get('episode_metrics', [])
    
    def clear(self):
        """Clear all collected metrics"""
        self.episode_metrics = []
        self.current_episode_measures = {}


def create_standard_measures(task_type: str, sim=None, task=None, config=None) -> List[Measure]:
    """
    Create a standard set of measures for a given task type
    
    Args:
        task_type: Type of task ("objectnav", "pointnav", etc.)
        sim: Simulator instance
        task: Task instance
        config: Configuration object
        
    Returns:
        List of configured measures
    """
    from .base_measures import (
        EpisodeStepsMeasure, 
        PathLengthMeasure, 
        PathEfficiencyMeasure,
        CollisionCountMeasure
    )
    
    measures = []
    
    # Common measures for all tasks
    measures.extend([
        EpisodeStepsMeasure(sim=sim, config=config, task=task),
        PathLengthMeasure(sim=sim, config=config, task=task),
        CollisionCountMeasure(sim=sim, config=config, task=task)
    ])
    
    # Task-specific measures
    if task_type.lower() == "objectnav":
        from .objectnav_measures import (
            ObjectNavSuccessMeasure,
            ObjectNavDistanceMeasure,
            ObjectFoundMeasure,
            ObjectCategoryMeasure
        )
        measures.extend([
            ObjectNavSuccessMeasure(sim=sim, config=config, task=task),
            ObjectNavDistanceMeasure(sim=sim, config=config, task=task),
            ObjectFoundMeasure(sim=sim, config=config, task=task),
            ObjectCategoryMeasure(sim=sim, config=config, task=task)
        ])
    
    elif task_type.lower() == "pointnav":
        from .pointnav_measures import (
            PointNavSuccessMeasure,
            PointNavDistanceMeasure,
            AngleToGoalMeasure,
            NavigationErrorMeasure
        )
        measures.extend([
            PointNavSuccessMeasure(sim=sim, config=config, task=task),
            PointNavDistanceMeasure(sim=sim, config=config, task=task),
            AngleToGoalMeasure(sim=sim, config=config, task=task),
            NavigationErrorMeasure(sim=sim, config=config, task=task)
        ])
    
    # Add path efficiency for tasks with known goals
    if task_type.lower() in ["objectnav", "pointnav"]:
        measures.append(PathEfficiencyMeasure(sim=sim, config=config, task=task))
    
    return measures


def print_metrics_summary(metrics: Dict[str, Any], title: str = "Metrics Summary"):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary of metrics to display
        title: Title for the summary
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    # Group metrics by category
    basic_metrics = {}
    performance_metrics = {}
    analysis_metrics = {}
    
    for key, value in metrics.items():
        if key in ['num_episodes', 'success_rate', 'completion_rate']:
            analysis_metrics[key] = value
        elif key in ['steps', 'distance_to_goal', 'path_length', 'time_to_goal']:
            performance_metrics[key] = value
        elif key != 'metrics':
            basic_metrics[key] = value
    
    # Print analysis metrics
    if analysis_metrics:
        print("\nOverall Analysis:")
        for key, value in analysis_metrics.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Print statistical metrics if available
    if 'metrics' in metrics:
        print("\nDetailed Statistics:")
        for metric_name, stats in metrics['metrics'].items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"  {metric_name}:")
                print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print(f"    Range: {stats['min']:.3f} - {stats['max']:.3f}")
                print(f"    Median: {stats['median']:.3f}")
    
    # Print other metrics
    if basic_metrics:
        print("\nOther Metrics:")
        for key, value in basic_metrics.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print()


def compare_metrics(metrics_list: List[Dict[str, Any]], labels: List[str] = None) -> Dict[str, Any]:
    """
    Compare metrics across different runs/conditions
    
    Args:
        metrics_list: List of metric dictionaries to compare
        labels: Optional labels for each metrics dict
        
    Returns:
        Comparison analysis
    """
    if not metrics_list:
        return {}
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(metrics_list))]
    
    comparison = {
        'runs': labels,
        'comparisons': {}
    }
    
    # Find common metrics
    common_metrics = set(metrics_list[0].keys())
    for metrics in metrics_list[1:]:
        common_metrics &= set(metrics.keys())
    
    # Compare each metric
    for metric_name in common_metrics:
        values = []
        for metrics in metrics_list:
            value = metrics[metric_name]
            if isinstance(value, (int, float, np.number)):
                values.append(float(value))
            else:
                values.append(value)
        
        if all(isinstance(v, (int, float)) for v in values):
            comparison['comparisons'][metric_name] = {
                'values': values,
                'best_run': labels[np.argmax(values) if 'success' in metric_name.lower() 
                                 else np.argmin(values)],
                'improvement': (max(values) - min(values)) / min(values) if min(values) > 0 else 0
            }
    
    return comparison