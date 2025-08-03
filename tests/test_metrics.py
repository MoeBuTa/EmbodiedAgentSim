"""Tests for metrics module functionality"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import easim
sys.path.insert(0, str(Path(__file__).parent.parent))

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import pytest
import numpy as np

from easim.metrics import (
    BaseMeasure,
    EpisodeStepsMeasure,
    PathLengthMeasure,
    ObjectNavSuccessMeasure,
    PointNavSuccessMeasure,
    MetricsCollector,
    create_standard_measures,
    analyze_metrics_batch,
    print_metrics_summary
)

from easim.tasks import ObjectNavTask


class TestBaseMeasures:
    """Test cases for base measures"""
    
    def test_episode_steps_measure(self):
        """Test EpisodeStepsMeasure functionality"""
        measure = EpisodeStepsMeasure(sim=None, config=None, task=None)
        
        # Test UUID
        assert measure.uuid == "episode_steps"
        
        # Test reset
        measure.reset_metric(episode=None)
        assert measure.get_metric() == 0
        
        # Test update
        measure.update_metric(episode=None, action=None, observations=None)
        assert measure.get_metric() == 1
        
        # Test multiple updates
        for i in range(5):
            measure.update_metric(episode=None, action=None, observations=None)
        assert measure.get_metric() == 6
    
    def test_path_length_measure(self):
        """Test PathLengthMeasure functionality"""
        measure = PathLengthMeasure(sim=None, config=None, task=None)
        
        assert measure.uuid == "path_length"
        
        # Test reset
        measure.reset_metric(episode=None)
        assert measure.get_metric() == 0.0
        
        # Test update (without simulator, metric should remain 0)
        measure.update_metric(episode=None, action=None, observations=None)
        assert measure.get_metric() == 0.0


class TestTaskSpecificMeasures:
    """Test cases for task-specific measures"""
    
    def test_objectnav_success_measure(self):
        """Test ObjectNavSuccessMeasure"""
        measure = ObjectNavSuccessMeasure(sim=None, config=None, task=None)
        
        assert measure.uuid == "objectnav_success"
        
        # Test reset
        measure.reset_metric(episode=None)
        assert measure.get_metric() == False
        
        # Test update with success (simulated)
        observations = {'objectgoal_sensor': np.array([0.1])}  # Close distance
        measure.update_metric(episode=None, action=None, observations=observations)
        assert measure.get_metric() == True
    
    def test_pointnav_success_measure(self):
        """Test PointNavSuccessMeasure"""
        measure = PointNavSuccessMeasure(sim=None, config=None, task=None)
        
        assert measure.uuid == "pointnav_success"
        
        # Test reset
        measure.reset_metric(episode=None)
        assert measure.get_metric() == False
        
        # Test update with success (simulated)
        observations = {'pointgoal_with_gps_compass': np.array([0.1, 0.0])}  # Close distance
        measure.update_metric(episode=None, action=None, observations=observations)
        assert measure.get_metric() == True


class TestMetricsCollector:
    """Test cases for MetricsCollector"""
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization"""
        collector = MetricsCollector()
        
        assert len(collector.measures) == 0
        assert len(collector.episode_metrics) == 0
        assert len(collector.current_episode_measures) == 0
    
    def test_metrics_collector_with_measures(self):
        """Test MetricsCollector with measures"""
        measures = [
            EpisodeStepsMeasure(sim=None, config=None, task=None),
            PathLengthMeasure(sim=None, config=None, task=None)
        ]
        collector = MetricsCollector(measures=measures)
        
        assert len(collector.measures) == 2
        
        # Test reset
        collector.reset_episode()
        assert len(collector.current_episode_measures) == 2
        
        # Test get current metrics
        current_metrics = collector.get_current_metrics()
        assert "episode_steps" in current_metrics
        assert "path_length" in current_metrics
    
    def test_metrics_collector_episode_workflow(self):
        """Test complete episode workflow with MetricsCollector"""
        measures = [EpisodeStepsMeasure(sim=None, config=None, task=None)]
        collector = MetricsCollector(measures=measures)
        
        # Start episode
        collector.reset_episode()
        
        # Take some steps
        for _ in range(3):
            collector.update_step()
        
        # End episode
        episode_metrics = collector.collect_episode(task=None)
        
        assert len(collector.episode_metrics) == 1
        assert "episode_steps" in episode_metrics
        
        # Test analysis
        analysis = collector.analyze_all_episodes()
        assert "num_episodes" in analysis
        assert analysis["num_episodes"] == 1


class TestMetricsUtils:
    """Test cases for metrics utility functions"""
    
    def test_analyze_metrics_batch(self):
        """Test analyze_metrics_batch function"""
        metrics_list = [
            {"success": True, "steps": 10, "distance": 2.5},
            {"success": False, "steps": 15, "distance": 5.0},
            {"success": True, "steps": 8, "distance": 1.2}
        ]
        
        analysis = analyze_metrics_batch(metrics_list)
        
        assert analysis["num_episodes"] == 3
        assert analysis["success_rate"] == 2/3
        assert "metrics" in analysis
        assert "steps" in analysis["metrics"]
        assert "distance" in analysis["metrics"]
        
        # Check statistics
        steps_stats = analysis["metrics"]["steps"]
        assert steps_stats["mean"] == 11.0  # (10 + 15 + 8) / 3
        assert steps_stats["min"] == 8
        assert steps_stats["max"] == 15
    
    def test_create_standard_measures(self):
        """Test create_standard_measures function"""
        # Test ObjectNav measures
        measures = create_standard_measures("objectnav")
        assert len(measures) > 0
        
        measure_uuids = [m.uuid for m in measures]
        assert "episode_steps" in measure_uuids
        assert "path_length" in measure_uuids
        
        # Test PointNav measures
        measures = create_standard_measures("pointnav")
        assert len(measures) > 0
        
        measure_uuids = [m.uuid for m in measures]
        assert "episode_steps" in measure_uuids
        assert "path_length" in measure_uuids
    
    def test_print_metrics_summary(self):
        """Test print_metrics_summary function"""
        metrics = {
            "success_rate": 0.75,
            "num_episodes": 4,
            "metrics": {
                "steps": {
                    "mean": 12.5,
                    "std": 2.1,
                    "min": 8,
                    "max": 16,
                    "median": 12.0
                }
            }
        }
        
        # This should not raise an exception
        print_metrics_summary(metrics, "Test Summary")


class TestMetricsIntegration:
    """Integration tests with actual tasks"""
    
    def test_metrics_with_objectnav_task(self):
        """Test metrics collection with ObjectNav task"""
        task = ObjectNavTask(dataset="HM3D", max_steps=5)
        
        try:
            # Create measures
            measures = create_standard_measures("objectnav", task=task)
            collector = MetricsCollector(measures=measures)
            
            # Run short episode
            collector.reset_episode()
            task.reset()
            
            for _ in range(2):
                collector.update_step()
                obs, done, info = task.step(1)  # Move forward
                if done:
                    break
            
            # Collect metrics
            episode_metrics = collector.collect_episode(task=task)
            
            assert "episode_steps" in episode_metrics
            assert isinstance(episode_metrics["episode_steps"], int)
            assert episode_metrics["episode_steps"] >= 2
            
        finally:
            task.close()


def test_metrics_module_imports():
    """Test that all metrics can be imported correctly"""
    from easim.metrics import (
        BaseMeasure,
        EpisodeStepsMeasure,
        ObjectNavSuccessMeasure,
        PointNavSuccessMeasure,
        MetricsCollector,
        Metrics,
        Measurements
    )
    
    # Verify classes exist
    assert BaseMeasure is not None
    assert EpisodeStepsMeasure is not None
    assert ObjectNavSuccessMeasure is not None
    assert PointNavSuccessMeasure is not None
    assert MetricsCollector is not None
    assert Metrics is not None
    assert Measurements is not None


if __name__ == "__main__":
    # Run basic tests manually
    print("Running metrics module tests...")
    
    test_base = TestBaseMeasures()
    test_base.test_episode_steps_measure()
    print("✓ EpisodeStepsMeasure test passed")
    
    test_task = TestTaskSpecificMeasures()
    test_task.test_objectnav_success_measure()
    print("✓ ObjectNavSuccessMeasure test passed")
    
    test_collector = TestMetricsCollector()
    test_collector.test_metrics_collector_initialization()
    print("✓ MetricsCollector initialization test passed")
    
    test_utils = TestMetricsUtils()
    test_utils.test_analyze_metrics_batch()
    print("✓ analyze_metrics_batch test passed")
    
    test_metrics_module_imports()
    print("✓ Metrics module imports test passed")
    
    print("\nAll metrics tests passed!")
    print("Run full test suite with: pytest tests/test_metrics.py -v")