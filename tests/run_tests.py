"""Test runner for EmbodiedAgentSim tests"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import easim
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_basic_tests():
    """Run basic functionality tests"""
    print("=" * 60)
    print("RUNNING BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    try:
        # Test video recorder
        print("\n1. Testing VideoRecorder...")
        from test_video_recorder import TestVideoRecorder, test_video_recorder_with_habitat_frames
        
        test_recorder = TestVideoRecorder()
        test_recorder.test_video_recorder_initialization()
        test_recorder.test_video_recorder_start_stop() 
        test_recorder.test_add_frame_rgb()
        test_video_recorder_with_habitat_frames()
        print("✓ VideoRecorder tests passed")
        
    except Exception as e:
        print(f"✗ VideoRecorder tests failed: {e}")
        return False
    
    try:
        # Test tasks
        print("\n2. Testing Task Classes...")
        from test_tasks import TestObjectNavTask, test_multiple_episodes
        
        test_task = TestObjectNavTask()
        test_task.test_objectnav_initialization_hm3d()
        test_task.test_action_processing()
        test_task.test_step_functionality()
        test_task.test_random_episode()
        test_multiple_episodes()
        print("✓ Task tests passed")
        
    except Exception as e:
        print(f"✗ Task tests failed: {e}")
        return False
    
    try:
        # Test integration
        print("\n3. Testing Integration...")
        from test_integration import TestTaskVideoIntegration, test_end_to_end_workflow
        
        test_integration = TestTaskVideoIntegration()
        test_integration.test_objectnav_with_video_recording()
        test_end_to_end_workflow()
        print("✓ Integration tests passed")
        
    except Exception as e:
        print(f"✗ Integration tests failed: {e}")
        return False
    
    return True


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("=" * 60)
    print("RUNNING QUICK SMOKE TEST")
    print("=" * 60)
    
    try:
        # Quick task initialization test
        print("Testing task initialization...")
        from easim.tasks import ObjectNavTask
        
        with ObjectNavTask(dataset="HM3D", max_steps=5) as task:
            observations = task.reset()
            obs, done, info = task.step("move_forward")
            stats = task.get_episode_stats()
        
        print("✓ Task initialization and basic step successful")
        
        # Quick video recorder test
        print("Testing video recorder...")
        from easim.utils.video_recorder import VideoRecorder
        from easim.utils.constants import OUTPUT_DIR
        import numpy as np
        
        video_path = OUTPUT_DIR / "smoke_test.mp4"
        recorder = VideoRecorder(str(video_path), fps=30)
        recorder.start_recording()
        
        # Add a test frame
        test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        recorder.add_frame(test_frame)
        
        recorder.stop_recording()
        
        assert video_path.exists()
        video_path.unlink()  # Clean up
        
        print("✓ Video recorder basic functionality successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    print("EmbodiedAgentSim Test Runner")
    print("Habitat-lab based embodied AI framework")
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--smoke":
            success = run_quick_smoke_test()
        elif sys.argv[1] == "--basic":
            success = run_basic_tests()
        else:
            print("Usage: python run_tests.py [--smoke|--basic]")
            print("  --smoke: Quick smoke test")
            print("  --basic: Full basic functionality tests")
            print("  (no args): Run smoke test by default")
            return
    else:
        # Default to smoke test
        success = run_quick_smoke_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The framework is working correctly.")
    else:
        print("❌ TESTS FAILED!")
        print("Please check the error messages above.")
        sys.exit(1)
    
    print("\nFor comprehensive testing, install pytest and run:")
    print("  pip install pytest")
    print("  pytest tests/ -v")


if __name__ == "__main__":
    main()