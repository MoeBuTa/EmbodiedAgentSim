"""Image recording functionality for habitat-lab framework"""
import numpy as np
from pathlib import Path
from PIL import Image

from easim.utils.constants import IMAGE_DIR


class ImageRecorder:
    """Records simulation images from RGB, depth, and semantic observations"""

    def __init__(self):
        self.frame_count = 0
        self.output_path = None

    def set_output_path(self, output_path: Path):
        """Set the base output path for images"""
        self.output_path = output_path

    def add_observations(self, observations: dict):
        """Add frames from observations and save as images"""
        from habitat_sim.utils.common import d3_40_colors_rgb
        
        if "rgb" in observations:
            rgb_frame = observations["rgb"]
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

    def _save_image(self, frame: np.ndarray, obs_type: str):
        """Save frame as image"""
        if self.output_path is None:
            return
            
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

    def reset(self):
        """Reset frame counter for new episode"""
        self.frame_count = 0