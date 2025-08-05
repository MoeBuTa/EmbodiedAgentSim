"""Tests for video recorder functionality"""
import sys
from pathlib import Path
import numpy as np

# Add the parent directory to the path so we can import easim
sys.path.insert(0, str(Path(__file__).parent.parent))

from easim.utils.video_recorder import VideoRecorder
from easim.utils.constants import OUTPUT_DIR


class TestVideoRecorder:
    """Test cases for VideoRecorder"""
    
    def test_video_recorder_initialization(self):
        """Test VideoRecorder initialization"""
        output_path = OUTPUT_DIR / "test_init.mp4"
        recorder = VideoRecorder(str(output_path), fps=30)
        
        assert recorder.output_path == output_path
        assert recorder.fps == 30
        assert recorder.writer is None
        assert len(recorder.frames) == 0
    
    def test_video_recorder_start_stop(self):
        """Test starting and stopping recording"""
        output_path = OUTPUT_DIR / "test_start_stop.mp4"
        recorder = VideoRecorder(str(output_path), fps=30)
        
        # Start recording
        recorder.start_recording()
        assert recorder.writer is not None
        
        # Stop recording
        recorder.stop_recording()
        assert recorder.writer is None
        
        # Check file was created
        assert output_path.exists()
        
        # Clean up
        output_path.unlink()
    
    def test_add_frame_rgb(self):
        """Test adding RGB frames"""
        output_path = OUTPUT_DIR / "test_add_frame_rgb.mp4"
        recorder = VideoRecorder(str(output_path), fps=30, resolution=(64, 48))
        
        recorder.start_recording()
        
        # Create test RGB frame
        rgb_frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        recorder.add_frame(rgb_frame)
        
        assert len(recorder.frames) == 1
        
        recorder.stop_recording()
        
        # Check file was created and has content
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Clean up
        output_path.unlink()
    
    def test_add_frame_different_formats(self):
        """Test adding frames in different formats"""
        output_path = OUTPUT_DIR / "test_formats.mp4"
        recorder = VideoRecorder(str(output_path), fps=30, resolution=(64, 48))
        
        recorder.start_recording()
        
        # Test float frame (0-1 range)
        float_frame = np.random.rand(48, 64, 3).astype(np.float32)
        recorder.add_frame(float_frame)
        
        # Test different size frame (should be resized)
        large_frame = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)
        recorder.add_frame(large_frame)
        
        # Test RGBA frame (should be converted to RGB)
        rgba_frame = np.random.randint(0, 255, (48, 64, 4), dtype=np.uint8)
        recorder.add_frame(rgba_frame)
        
        assert len(recorder.frames) == 3
        
        recorder.stop_recording()
        
        # Check file was created
        assert output_path.exists()
        
        # Clean up
        output_path.unlink()

    
    def test_none_frame_handling(self):
        """Test that None frames are handled gracefully"""
        output_path = OUTPUT_DIR / "test_none.mp4"
        recorder = VideoRecorder(str(output_path), fps=30)
        
        recorder.start_recording()
        
        # Add None frame (should be ignored)
        recorder.add_frame(None)
        
        # Add normal frame
        normal_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        recorder.add_frame(normal_frame)
        
        # Should only have one frame
        assert len(recorder.frames) == 1
        
        recorder.stop_recording()
        
        # Clean up
        output_path.unlink()


def test_video_recorder_with_habitat_frames():
    """Test video recorder with realistic habitat-lab frame dimensions"""
    output_path = OUTPUT_DIR / "test_habitat_frames.mp4"
    recorder = VideoRecorder(str(output_path), fps=30)
    
    recorder.start_recording()
    
    # Simulate habitat-lab RGB observations (typical size: 256x256x3)
    for i in range(5):
        # Create frame with some pattern
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 50  # Red channel
        frame[:, :, 1] = (255 - i * 50)  # Green channel
        frame[:, :, 2] = 128  # Blue channel
        
        recorder.add_frame(frame)
    
    recorder.stop_recording()
    
    # Verify file creation and content
    assert output_path.exists()
    assert output_path.stat().st_size > 1000  # Should have substantial content
    
    # Clean up
    output_path.unlink()


if __name__ == "__main__":
    # Run a simple test manually
    print("Running basic video recorder tests...")
    
    test_recorder = TestVideoRecorder()
    test_recorder.test_video_recorder_initialization()
    print("✓ Video recorder initialization test passed")
    
    test_recorder.test_video_recorder_start_stop()
    print("✓ Start/stop recording test passed")
    
    test_recorder.test_add_frame_rgb()
    print("✓ Add RGB frame test passed")
    
    test_video_recorder_with_habitat_frames()
    print("✓ Habitat-lab frame test passed")
    
    print("\nAll video recorder tests passed!")
    print("Run: pytest tests/test_video_recorder.py -v")