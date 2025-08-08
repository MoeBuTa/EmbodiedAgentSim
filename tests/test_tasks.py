"""Tests for task classes"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import easim
sys.path.insert(0, str(Path(__file__).parent.parent))

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import pytest
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np

from easim.tasks import BaseEmbodiedTask, ObjectNavTask
from easim.recorders.video import VideoRecorder
from easim.utils.constants import OUTPUT_DIR


class TestBaseEmbodiedTask:
    """Test cases for BaseEmbodiedTask"""
    
    def test_task_initialization(self):
        """Test that ObjectNavTask initializes correctly"""
        task = ObjectNavTask(dataset="HM3D", max_steps=50)
        
        assert task.dataset == "HM3D"
        assert task.max_steps == 50
        assert task.env is not None
        assert task.step_count == 0
        
        task.close()
    
    def test_task_reset(self):
        """Test task reset functionality"""
        task = ObjectNavTask(dataset="HM3D", max_steps=50)
        
        observations = task.reset()
        
        assert observations is not None
        assert isinstance(observations, dict)
        assert task.step_count == 0
        assert 'rgb' in observations
        
        task.close()
    
    def test_context_manager(self):
        """Test that task works as context manager"""
        with ObjectNavTask(dataset="HM3D", max_steps=10) as task:
            observations = task.reset()
            assert observations is not None
            assert task.env is not None
        
        # Should be closed after context exit
        assert task.env is None


class TestObjectNavTask:
    """Test cases for ObjectNavTask"""
    
    def test_objectnav_initialization_hm3d(self):
        """Test ObjectNavTask initialization with HM3D dataset"""
        task = ObjectNavTask(dataset="HM3D", max_steps=100)
        
        assert task.dataset == "HM3D"
        assert "objectnav_hm3d" in task.config_path
        assert task.max_steps == 100
        
        task.close()
    
    def test_objectnav_initialization_mp3d(self):
        """Test ObjectNavTask initialization with MP3D dataset"""
        task = ObjectNavTask(dataset="MP3D", max_steps=100)
        
        assert task.dataset == "MP3D"
        assert "objectnav_mp3d" in task.config_path
        
        task.close()
    
    def test_invalid_dataset(self):
        """Test that invalid dataset raises ValueError"""
        with pytest.raises(ValueError):
            ObjectNavTask(dataset="INVALID")
    
    def test_action_processing(self):
        """Test action processing functionality"""
        task = ObjectNavTask(dataset="HM3D", max_steps=10)
        
        # Test enum action
        action = task._process_action(HabitatSimActions.move_forward)
        assert action == HabitatSimActions.move_forward
        
        # Test string action
        action = task._process_action("move_forward")
        assert action == HabitatSimActions.move_forward
        
        # Test integer action
        action = task._process_action(1)
        assert action == HabitatSimActions.move_forward
        
        task.close()
    
    def test_step_functionality(self):
        """Test basic step functionality"""
        task = ObjectNavTask(dataset="HM3D", max_steps=5)
        
        observations = task.reset()
        initial_step_count = task.step_count
        
        obs, done, info = task.step(HabitatSimActions.move_forward)
        
        assert task.step_count == initial_step_count + 1
        assert isinstance(obs, dict)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'step' in info
        
        task.close()
    
    def test_observation_access(self):
        """Test observation access methods"""
        task = ObjectNavTask(dataset="HM3D", max_steps=10)
        
        observations = task.reset()
        
        # Test RGB observation
        rgb_obs = task.get_rgb_observation()
        assert rgb_obs is not None
        assert isinstance(rgb_obs, np.ndarray)
        
        # Test current observations
        current_obs = task.get_observations()
        assert current_obs is not None
        assert isinstance(current_obs, dict)
        
        task.close()
    
    def test_task_info(self):
        """Test task information retrieval"""
        task = ObjectNavTask(dataset="HM3D", max_steps=100, success_distance=0.15)
        
        task_info = task.get_task_info()
        
        assert task_info['task_type'] == 'ObjectNav'
        assert task_info['dataset'] == 'HM3D'
        assert task_info['max_steps'] == 100
        assert task_info['success_distance'] == 0.15
        
        task.close()
    
    def test_random_episode(self):
        """Test running a complete random episode"""
        task = ObjectNavTask(dataset="HM3D", max_steps=20)
        
        result = task.run_random_episode()
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'steps' in result
        assert 'task_type' in result
        assert result['steps'] <= 20
        
        task.close()


class TestVideoRecorderIntegration:
    """Test integration with video recorder"""
    
    def test_task_with_video_recording(self):
        """Test that task works with video recording"""
        task = ObjectNavTask(dataset="HM3D", max_steps=10)
        
        video_path = OUTPUT_DIR / "test_recording.mp4"
        recorder = VideoRecorder(fps=30)
        recorder.start_recording(str(video_path))
        
        try:
            observations = task.reset()
            
            for _ in range(5):
                rgb_obs = task.get_rgb_observation()
                if rgb_obs is not None:
                    recorder.add_frame(rgb_obs)
                
                obs, done, info = task.step(HabitatSimActions.move_forward)
                
                if done:
                    break
            
            recorder.stop_recording()
            
            # Check that video file was created
            assert video_path.exists()
            assert video_path.stat().st_size > 0
            
        finally:
            task.close()
            # Clean up test video
            if video_path.exists():
                video_path.unlink()


def test_multiple_episodes():
    """Test running multiple episodes for statistics"""
    task = ObjectNavTask(dataset="HM3D", max_steps=10)
    
    try:
        results = []
        num_episodes = 3
        
        for _ in range(num_episodes):
            result = task.run_random_episode()
            results.append(result)
        
        assert len(results) == num_episodes
        
        # Check that all results have required fields
        for result in results:
            assert 'success' in result
            assert 'steps' in result
            assert 'task_type' in result
            assert isinstance(result['success'], bool)
            assert isinstance(result['steps'], int)
            assert result['steps'] <= 10
        
        # Calculate basic statistics
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_steps = np.mean([r['steps'] for r in results])
        
        assert 0 <= success_rate <= 1
        assert 0 <= avg_steps <= 10
        
    finally:
        task.close()


if __name__ == "__main__":
    # Run a simple test manually
    print("Running basic task functionality test...")
    
    test_task = TestObjectNavTask()
    test_task.test_objectnav_initialization_hm3d()
    print("✓ HM3D initialization test passed")
    
    test_task.test_step_functionality()
    print("✓ Step functionality test passed")
    
    test_task.test_random_episode()
    print("✓ Random episode test passed")
    
    print("\nAll basic tests passed! Use pytest for full test suite.")
    print("Run: pytest tests/test_tasks.py -v")