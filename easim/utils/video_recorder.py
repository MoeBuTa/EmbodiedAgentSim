"""Video recording functionality for habitat-lab framework"""
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path

from easim.utils.constants import (
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_CODEC
)


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