"""Integration tests for complete workflows"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import easim
sys.path.insert(0, str(Path(__file__).parent.parent))

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np

from easim.tasks import ObjectNavTask
from easim.utils.video_recorder import VideoRecorder
from easim.utils.constants import OUTPUT_DIR


class TestTaskVideoIntegration:
    """Integration tests for tasks with video recording"""
    
    def test_objectnav_with_video_recording(self):
        """Test complete ObjectNav workflow with video recording"""
        # Setup
        task = ObjectNavTask(dataset="HM3D", max_steps=10)
        video_path = OUTPUT_DIR / "integration_test.mp4"
        recorder = VideoRecorder(str(video_path), fps=30)
        
        try:
            # Start recording
            recorder.start_recording()
            
            # Reset task
            observations = task.reset()
            assert observations is not None
            
            # Get task info
            task_info = task.get_task_info()
            print(f"Testing {task_info['task_type']} on {task_info['dataset']}")
            print(f"Target object: {task.get_target_object()}")
            
            # Run episode with recording
            step_count = 0
            while not task._is_episode_done() and step_count < 5:
                # Record frame
                rgb_obs = task.get_rgb_observation()
                if rgb_obs is not None:
                    recorder.add_frame(rgb_obs)
                
                # Take random action
                if np.random.random() < 0.6:
                    action = HabitatSimActions.move_forward
                else:
                    action = np.random.choice([
                        HabitatSimActions.turn_left,
                        HabitatSimActions.turn_right
                    ])
                
                obs, done, info = task.step(action)
                step_count += 1
                
                # Verify step info
                assert 'step' in info
                assert info['step'] == step_count
                assert 'target_object' in info
            
            # Stop recording
            recorder.stop_recording()
            
            # Verify results
            episode_stats = task.get_episode_stats()
            assert episode_stats['steps'] == step_count
            assert isinstance(episode_stats['success'], bool)
            
            # Verify video file
            assert video_path.exists()
            assert video_path.stat().st_size > 0
            
            print(f"Episode completed: {episode_stats['steps']} steps, Success: {episode_stats['success']}")
            
        finally:
            task.close()
            # Clean up
            if video_path.exists():
                video_path.unlink()
    
    def test_multiple_episodes_with_statistics(self):
        """Test running multiple episodes and collecting statistics"""
        task = ObjectNavTask(dataset="HM3D", max_steps=15)
        
        try:
            results = []
            num_episodes = 3
            
            for episode in range(num_episodes):
                print(f"Running episode {episode + 1}/{num_episodes}")
                
                result = task.run_random_episode()
                results.append(result)
                
                # Verify result structure
                assert 'success' in result
                assert 'steps' in result
                assert 'task_type' in result
                assert 'target_object' in result
                assert result['steps'] <= 15
                
                print(f"  Target: {result['target_object']}, Steps: {result['steps']}, Success: {result['success']}")
            
            # Calculate statistics
            success_rate = sum(r['success'] for r in results) / len(results)
            avg_steps = np.mean([r['steps'] for r in results])
            
            print(f"\nStatistics:")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average steps: {avg_steps:.1f}")
            
            # Verify statistics are reasonable
            assert 0 <= success_rate <= 1
            assert 0 <= avg_steps <= 15
            
        finally:
            task.close()
    
    def test_task_context_manager_with_recording(self):
        """Test using task as context manager with video recording"""
        video_path = OUTPUT_DIR / "context_test.mp4"
        
        try:
            with ObjectNavTask(dataset="HM3D", max_steps=5) as task:
                recorder = VideoRecorder(str(video_path))
                recorder.start_recording()
                
                observations = task.reset()
                
                # Take a few steps
                for _ in range(3):
                    rgb_obs = task.get_rgb_observation()
                    if rgb_obs is not None:
                        recorder.add_frame(rgb_obs)
                    
                    obs, done, info = task.step(HabitatSimActions.move_forward)
                    if done:
                        break
                
                recorder.stop_recording()
                
                # Verify task is still functional
                assert task.env is not None
                episode_stats = task.get_episode_stats()
                assert episode_stats['steps'] > 0
            
            # Task should be closed after context exit
            assert task.env is None
            
            # Video should still exist
            assert video_path.exists()
            
        finally:
            if video_path.exists():
                video_path.unlink()


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("Running end-to-end integration test...")
    
    # Initialize components
    task = ObjectNavTask(dataset="HM3D", max_steps=8, success_distance=0.2)
    video_path = OUTPUT_DIR / "end_to_end_test.mp4"
    
    try:
        recorder = VideoRecorder(str(video_path), fps=10)  # Lower FPS for test
        recorder.start_recording()
        
        # Reset and get initial state
        observations = task.reset()
        task_info = task.get_task_info()
        initial_distance = task.get_distance_to_goal()
        
        print(f"Task: {task_info['task_type']}")
        print(f"Target: {task.get_target_object()}")
        if initial_distance:
            print(f"Initial distance: {initial_distance:.2f}m")
        
        # Run episode
        actions_taken = []
        distances = []
        
        while not task._is_episode_done():
            # Record current state
            rgb_obs = task.get_rgb_observation()
            if rgb_obs is not None:
                recorder.add_frame(rgb_obs)
            
            current_distance = task.get_distance_to_goal()
            if current_distance:
                distances.append(current_distance)
            
            # Choose action (simple policy)
            if len(distances) >= 2 and distances[-1] < distances[-2]:
                # Distance decreasing, continue forward
                action = HabitatSimActions.move_forward
            elif np.random.random() < 0.5:
                action = HabitatSimActions.move_forward
            else:
                action = np.random.choice([
                    HabitatSimActions.turn_left,
                    HabitatSimActions.turn_right
                ])
            
            actions_taken.append(action)
            obs, done, info = task.step(action)
        
        # Finalize
        recorder.stop_recording()
        final_stats = task.get_episode_stats()
        
        # Results summary
        print(f"\nResults:")
        print(f"  Steps taken: {final_stats['steps']}")
        print(f"  Actions: {[str(a).split('.')[-1] for a in actions_taken]}")
        print(f"  Success: {final_stats['success']}")
        print(f"  Final distance: {task.get_distance_to_goal():.2f}m" if task.get_distance_to_goal() else "Unknown")
        print(f"  Video saved: {video_path}")
        
        # Verify everything worked
        assert len(actions_taken) == final_stats['steps']
        assert video_path.exists()
        assert video_path.stat().st_size > 0
        
        print("✓ End-to-end test completed successfully!")
        
    finally:
        task.close()
        # Clean up
        if video_path.exists():
            video_path.unlink()


if __name__ == "__main__":
    # Run integration tests manually
    print("Running integration tests...")
    
    test_integration = TestTaskVideoIntegration()
    test_integration.test_objectnav_with_video_recording()
    print("✓ ObjectNav with video recording test passed")
    
    test_integration.test_multiple_episodes_with_statistics()
    print("✓ Multiple episodes test passed")
    
    test_end_to_end_workflow()
    
    print("\nAll integration tests passed!")
    print("Run full test suite with: pytest tests/ -v")