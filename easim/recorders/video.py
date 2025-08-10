"""Video recording functionality for habitat-lab framework"""
import cv2
import numpy as np
from typing import Tuple, Dict

from easim.utils.constants import (
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_CODEC
)


class VideoRecorder:
    """Records simulation videos from RGB observations only"""

    def __init__(self, fps: int = DEFAULT_FPS,
                 resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION):
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.frames = []
        self.run_name = None
        self.episode_name = None
        self.output_path = None

    def set_run_and_episode(self, run_name: str, episode_name: str):
        """Set run and episode names for proper directory structure"""
        self.run_name = run_name
        self.episode_name = episode_name
        
    def start_recording(self, output_path: str = None):
        """Initialize video writer for RGB"""
        from easim.utils.constants import VIDEO_DIR
        
        if self.run_name and self.episode_name:
            # Create directory structure: videos/run_xxx/episode_xxx.mp4
            video_dir = VIDEO_DIR / self.run_name
            video_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = video_dir / f"{self.episode_name}.mp4"
        else:
            self.output_path = output_path
            
        if self.output_path is None:
            return
            
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.resolution
        )
        self.frames = []

    def add_frame(self, rgb_frame: np.ndarray):
        """Add RGB frame to video"""
        # Resize if needed
        if rgb_frame.shape[:2] != self.resolution[::-1]:
            rgb_frame = cv2.resize(rgb_frame, self.resolution)
        
        # Convert to BGR for OpenCV video writer
        frame_bgr = cv2.cvtColor(rgb_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        self.writer.write(frame_bgr)
        self.frames.append(frame_bgr.copy())

    def stop_recording(self):
        """Finalize and save RGB video"""
        if self.writer:
            self.writer.release()
            self.writer = None
        self.frames.clear()