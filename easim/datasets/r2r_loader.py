import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from easim.utils.constants import DATA_PATH, PROJECT_DIR


@dataclass
class R2REpisode:
    """R2R episode data structure"""
    episode_id: str
    scan: str  # Scene name
    heading: float  # Starting heading
    instructions: List[str]  # List of instruction variants
    path: List[List[float]]  # Ground truth path (positions)
    distance: float  # Path length


class R2RDatasetLoader:
    """Loader for Room-to-Room (R2R) dataset"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_PATH / "datasets" / "vln" / "r2r"
        self.episodes = {}
        self.splits = ["train", "val_seen", "val_unseen", "test"]

    def download_r2r_data(self):
        """Download R2R dataset files"""
        print("Downloading R2R dataset...")

        # Create directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # URLs for R2R data
        base_url = "https://github.com/peteanderson80/Matterport3DSimulator/raw/master/tasks/R2R/data"
        files = {
            "train": "R2R_train.json",
            "val_seen": "R2R_val_seen.json",
            "val_unseen": "R2R_val_unseen.json",
            "test": "R2R_test.json"
        }

        import urllib.request

        for split, filename in files.items():
            url = f"{base_url}/{filename}"
            output_path = self.data_dir / filename

            if not output_path.exists():
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, output_path)
                    print(f"✅ Downloaded {filename}")
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
            else:
                print(f"✅ {filename} already exists")

    def load_split(self, split: str) -> List[R2REpisode]:
        """Load episodes from a specific split"""
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {self.splits}")

        # Download data if needed
        if not self.data_dir.exists():
            self.download_r2r_data()

        # Load JSON file
        filename = f"R2R_{split}.json"
        filepath = self.data_dir / filename

        if not filepath.exists():
            print(f"File not found: {filepath}")
            self.download_r2r_data()

        if not filepath.exists():
            raise FileNotFoundError(f"Could not find or download {filepath}")

        print(f"Loading R2R {split} split from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        episodes = []
        for item in data:
            episode = R2REpisode(
                episode_id=item["path_id"],
                scan=item["scan"],
                heading=item["heading"],
                instructions=item["instructions"],
                path=item["path"],
                distance=item.get("distance", 0.0)
            )
            episodes.append(episode)

        print(f"✅ Loaded {len(episodes)} episodes from {split} split")
        return episodes

    def get_episode_by_id(self, episode_id: str, split: str) -> Optional[R2REpisode]:
        """Get a specific episode by ID"""
        episodes = self.load_split(split)
        for episode in episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def get_scenes_used(self, split: str) -> List[str]:
        """Get list of scenes used in a split"""
        episodes = self.load_split(split)
        scenes = list(set(episode.scan for episode in episodes))
        return sorted(scenes)

    def print_dataset_stats(self):
        """Print statistics about the R2R dataset"""
        print("\n=== R2R Dataset Statistics ===")

        for split in self.splits:
            try:
                episodes = self.load_split(split)
                scenes = self.get_scenes_used(split)

                # Calculate instruction statistics
                total_instructions = sum(len(ep.instructions) for ep in episodes)
                avg_instructions = total_instructions / len(episodes) if episodes else 0

                # Calculate path statistics
                path_lengths = [ep.distance for ep in episodes if ep.distance > 0]
                avg_path_length = np.mean(path_lengths) if path_lengths else 0

                print(f"\n{split.upper()}:")
                print(f"  Episodes: {len(episodes)}")
                print(f"  Scenes: {len(scenes)}")
                print(f"  Instructions: {total_instructions}")
                print(f"  Avg instructions/episode: {avg_instructions:.1f}")
                print(f"  Avg path length: {avg_path_length:.1f}m")

            except Exception as e:
                print(f"\n{split.upper()}: ❌ Failed to load ({e})")


def download_r2r_scenes():
    """Download MP3D scenes used in R2R"""
    print("Getting R2R scene list...")

    loader = R2RDatasetLoader()

    # Get all scenes used in R2R
    all_scenes = set()
    for split in ["train", "val_seen", "val_unseen"]:
        try:
            scenes = loader.get_scenes_used(split)
            all_scenes.update(scenes)
            print(f"{split}: {len(scenes)} scenes")
        except Exception as e:
            print(f"Failed to load {split}: {e}")

    print(f"\nTotal unique scenes in R2R: {len(all_scenes)}")
    print("Scene list:", sorted(all_scenes))

    # Instructions for downloading
    print(f"\nTo download these {len(all_scenes)} scenes:")
    print("1. Download full MP3D dataset (large ~15GB):")
    print("   python -m habitat_sim.utils.datasets_download --uids mp3d --data-path data")
    print("\n2. Or manually download specific scenes from:")
    print("   https://niessner.github.io/Matterport/")

    return sorted(all_scenes)


if __name__ == "__main__":
    # Example usage
    loader = R2RDatasetLoader()

    # Download and show stats
    loader.print_dataset_stats()

    # Show scenes needed
    download_r2r_scenes()