"""Image recording functionality for habitat-lab framework"""
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from easim.utils.constants import IMAGE_DIR


class ImageRecorder:
    """Records simulation images from RGB, depth, and semantic observations"""

    def __init__(self):
        self.frame_count = 0
        self.output_path = None
        self.task_name = None
        self.run_name = None
        self.episode_name = None

    def set_output_path(self, output_path: Path):
        """Set the base output path for images"""
        self.output_path = output_path
        
    def set_task_run_and_episode(self, task_name: str, run_name: str, episode_name: str):
        """Set task, run and episode names for proper directory structure"""
        self.task_name = task_name
        self.run_name = run_name
        self.episode_name = episode_name

    def add_observations(self, observations: dict):
        """Add frames from observations and save as images"""
        from habitat_sim.utils.common import d3_40_colors_rgb
        
        if "rgb" in observations:
            rgb_frame = observations["rgb"]
            self._save_image(rgb_frame, "rgb")

        if "depth" in observations:
            depth_data = observations["depth"]
            if depth_data.ndim == 3:
                depth_data = depth_data.squeeze(-1)
            
            # Normalize depth values to 0-1 range
            depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)
            
            # Apply colormap (using 'plasma' which gives purple-to-yellow similar to your image)
            colormap = cm.get_cmap('plasma')
            depth_colored = colormap(depth_normalized)
            
            # Convert to 0-255 RGB format
            depth_frame = (depth_colored[:, :, :3] * 255).astype(np.uint8)
            self._save_image(depth_frame, "depth")

        if "semantic" in observations:
            semantic_data = observations["semantic"].flatten() % 40
            semantic_rgb = d3_40_colors_rgb[semantic_data].reshape(
                observations["semantic"].shape[0], observations["semantic"].shape[1], 3
            )
            self._save_image(semantic_rgb, "semantic")

        self.frame_count += 1

    def _save_image(self, frame: np.ndarray, obs_type: str):
        """Save frame as image"""
        if self.task_name is None or self.run_name is None or self.episode_name is None:
            return
            
        # Create directory structure: images/{task_name}/run_{}/episode_{num}_{id}/
        images_dir = IMAGE_DIR / self.task_name / self.run_name / self.episode_name
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"step_{self.frame_count:04d}_{obs_type}.jpg"
        Image.fromarray(frame.astype(np.uint8)).save(str(image_path))

    def reset(self):
        """Reset frame counter for new episode"""
        self.frame_count = 0