"""Video recording functionality for habitat-lab framework"""
import cv2
import numpy as np
from typing import Tuple, Dict, TYPE_CHECKING
from pathlib import Path

from easim.utils.constants import (
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_CODEC, VIDEO_DIR
)

if TYPE_CHECKING:
    from habitat import Agent


class VideoRecorder:
    """Records simulation videos from RGB observations"""

    def __init__(self, output_path: str, fps: int = DEFAULT_FPS,
                 resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION):
        self.output_path = Path(output_path)
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.frames = []

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def start_recording(self):
        """Initialize video writer"""
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.resolution
        )
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video with proper RGB->BGR conversion"""
        if frame is None:
            return

        # Ensure proper data type
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:  # OpenCV uses (height, width)
            frame = cv2.resize(frame, self.resolution)

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Write frame
        if self.writer is not None:
            self.writer.write(frame)

        self.frames.append(frame.copy())

    def stop_recording(self):
        """Finalize and save video"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print(f"Video saved to: {self.output_path}")

    def save_frames_as_images(self, output_dir: str):
        """Save individual frames as images"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame_path = output_path / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)

    @staticmethod
    def setup_video_directory(task_name: str) -> Path:
        """
        Set up video recording directory structure.
        
        :param task_name: Name of the task/benchmark for directory naming.
        :return: Path to the video directory for this evaluation run.
        """
        # Find next incremental number for this agent-task combination
        base_dir = VIDEO_DIR / f"{task_name}"
        run_number = 1
        while (base_dir / f"run_{run_number:03d}").exists():
            run_number += 1
        
        video_dir = base_dir / f"run_{run_number:03d}"
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    @staticmethod
    def record_episode_with_video(env, agent: "Agent", episode_num: int, video_dir: Path) -> Dict:
        """
        Record a single episode with video recording.
        
        :param env: The habitat environment.
        :param agent: The agent to evaluate.
        :param episode_num: Episode number for naming the video file.
        :param video_dir: Directory to save the video file.
        :return: Dictionary containing episode metrics.
        """
        observations = env.reset()
        agent.reset()
        
        # Initialize video recorder for this episode
        video_path = video_dir / f"episode_{episode_num + 1:03d}.mp4"
        video_recorder = VideoRecorder(str(video_path))
        video_recorder.start_recording()
        
        try:
            while not env.episode_over:
                # Record frame if RGB observations are available
                if "rgb" in observations:
                    video_recorder.add_frame(observations["rgb"])
                
                action = agent.act(observations)
                observations = env.step(action)
        finally:
            # Ensure video recording stops even if an error occurs
            video_recorder.stop_recording()
        
        return env.get_metrics()

    @staticmethod
    def record_episode_no_video(env, agent: "Agent") -> Dict:
        """
        Record a single episode without video recording.
        
        :param env: The habitat environment.
        :param agent: The agent to evaluate.
        :return: Dictionary containing episode metrics.
        """
        observations = env.reset()
        agent.reset()
        
        while not env.episode_over:
            action = agent.act(observations)
            observations = env.step(action)
        
        return env.get_metrics()