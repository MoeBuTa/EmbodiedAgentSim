"""Video recording functionality for habitat-lab framework"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Any
from pathlib import Path
from PIL import Image

from easim.utils.constants import (
    DEFAULT_FPS, DEFAULT_VIDEO_RESOLUTION, DEFAULT_VIDEO_CODEC, VIDEO_DIR, IMAGE_DIR
)

from habitat import Agent


def get_episode_filename(episode_num: int, scene_id: str) -> str:
    """
    Generate episode filename with incremental number and scene identifier.
    
    :param episode_num: Episode number (0-based, will be converted to 1-based)
    :param scene_id: Scene ID path from the episode
    :return: Filename string like "episode_1_00800-TEEsavR23oF"
    """
    # Extract scene identifier from path
    if '/' in scene_id:
        # Extract folder name which contains the scene identifier
        # e.g., "hm3d_v0.2/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb" -> "00800-TEEsavR23oF"
        scene_identifier = scene_id.split('/')[-2]
    else:
        # Fallback for direct scene names
        scene_identifier = scene_id.split('.')[0]
    
    # Use episode_num + 1 for incremental naming (episode_1, episode_2, etc.)
    return f"episode_{episode_num + 1}_{scene_identifier}"


class VideoRecorder:
    """Records simulation videos from RGB, depth, and semantic observations"""

    def __init__(self, output_path: Path, fps: int = DEFAULT_FPS,
                 resolution: Tuple[int, int] = DEFAULT_VIDEO_RESOLUTION):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writers = {}  # Multiple writers for different observation types
        self.frames = {}  # Store frames by observation type
        self.frame_count = 0  # Counter for image naming
        self.images_dir = None  # Will be set if image saving is enabled

    def start_recording(self, available_obs_types=None):
        """Initialize video writer for RGB only"""
        if "rgb" in available_obs_types:
            fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
            
            self.writers["rgb"] = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.resolution
            )
            self.frames["rgb"] = []

    def add_observations(self, observations: Dict, save_images: bool = True):
        """Add frames from observations - video for RGB, images for all modalities"""
        from habitat_sim.utils.common import d3_40_colors_rgb
        
        if "rgb" in observations:
            rgb_frame = observations["rgb"]
            # Record RGB to video
            self._record_video_frame(rgb_frame, "rgb")
            # Save RGB image
            self._save_image(rgb_frame, "rgb")

        if "depth" in observations:
            depth_data = (observations["depth"] / 10 * 255).astype(np.uint8)
            if depth_data.ndim == 3:
                depth_data = depth_data.squeeze(-1)
            depth_frame = np.stack([depth_data] * 3, axis=-1)
            self._save_image(depth_frame, "depth")

        if "semantic" in observations:
            semantic_data = observations["semantic"].flatten() % 40
            semantic_rgb = d3_40_colors_rgb[semantic_data].reshape(
                observations["semantic"].shape[0], observations["semantic"].shape[1], 3
            )
            self._save_image(semantic_rgb, "semantic")

        self.frame_count += 1

    def _record_video_frame(self, frame: np.ndarray, obs_type: str):
        """Record frame to video only"""
        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)
        
        # Convert to BGR for OpenCV video writer
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

        self.writers[obs_type].write(frame_bgr)
        self.frames[obs_type].append(frame_bgr.copy())
    
    def _save_image(self, frame: np.ndarray, obs_type: str):
        """Save frame as image"""
        # Extract task_name, run_number, episode_id from output_path
        # Path structure: data/output/videos/task_name/run_XXX/episode_{episode_id}.mp4
        path_parts = self.output_path.parts
        task_name = path_parts[-3]  # task_name
        run_number = path_parts[-2]  # run_XXX
        episode_name = self.output_path.stem  # episode_{episode_id}

        images_dir = IMAGE_DIR / task_name / run_number / episode_name
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"step_{self.frame_count:04d}_{obs_type}.jpg"
        Image.fromarray(frame.astype(np.uint8)).save(str(image_path))


    def stop_recording(self):
        """Finalize and save RGB video"""
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()

        if "rgb" in self.frames:
            print(f"RGB video saved to: {self.output_path}")

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
    def record_episode_with_video(env, agent: "Agent", episode_num: int, video_dir: Path) -> Optional[Any]:
        """
        Record a single episode with video recording.
        
        :param env: The habitat environment.
        :param agent: The agent to evaluate.
        :param episode_num: Episode number for naming the video file.
        :param video_dir: Directory to save the video file.
        :return: Dictionary containing episode metrics.
        """
        # Reset environment to get next episode from the dataset
        # Habitat automatically advances to the next episode via EpisodeIterator
        observations = env.reset()
        agent.reset()

        available_obs_types = [obs_type for obs_type in ["rgb", "depth", "semantic"] if obs_type in observations]

        # Initialize video recorder for this episode
        # Get episode info after reset to ensure we have the current episode
        episode_filename = get_episode_filename(episode_num, env.current_episode.scene_id)
        video_path = video_dir / f"{episode_filename}.mp4"
        video_recorder = VideoRecorder(video_path)
        video_recorder.start_recording(available_obs_types)

        try:
            while not env.episode_over:
                # Record observations if any of rgb, depth, semantic are available
                video_recorder.add_observations(observations)

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
