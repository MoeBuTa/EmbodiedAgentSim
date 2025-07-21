"""
Room-to-Room (R2R) Vision-Language Navigation dataset
"""
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base.dataset import FileBasedDataset, DatasetConfig
from ...tasks.base.episode import VLNEpisode, Position, Goal
from ...utils.constants import R2R_DATASET_DIR, R2R_SPLITS


class R2RDataset(FileBasedDataset):
    """Room-to-Room Vision-Language Navigation dataset"""

    def __init__(self,
                 split: str = "val_seen",
                 max_episodes: Optional[int] = None,
                 data_path: Optional[str] = None,
                 instruction_variant: str = "all",  # "all", "first", "random"
                 **kwargs):

        # Setup paths
        if data_path is None:
            data_path = R2R_DATASET_DIR

        config = DatasetConfig(
            name=f"r2r_{split}",
            split=split,
            data_path=str(data_path),
            max_episodes=max_episodes,
            **kwargs
        )

        self.data_path = Path(data_path)
        self.instruction_variant = instruction_variant

        # Get data files
        data_files = [self._get_episode_file()]

        super().__init__(config, data_files)

    def load_episodes(self):
        """Load R2R episodes"""
        episode_file = self._get_episode_file()

        if not episode_file.exists():
            self._download_dataset()

        if not episode_file.exists():
            print(f"Episode file not found: {episode_file}")
            print("Please check R2R dataset download")
            return

        print(f"Loading R2R episodes from {episode_file}")

        # Load episodes from file
        with open(episode_file, 'r') as f:
            data = json.load(f)

        episodes = []

        # Apply max_episodes limit
        if self.config.max_episodes:
            data = data[:self.config.max_episodes]

        for i, ep_data in enumerate(data):
            try:
                # R2R can have multiple instruction variants per path
                if self.instruction_variant == "all":
                    # Create separate episode for each instruction
                    for j, instruction in enumerate(ep_data["instructions"]):
                        episode = self._parse_episode(ep_data, instruction, f"{i}_{j}")
                        episodes.append(episode)
                elif self.instruction_variant == "first":
                    # Use first instruction only
                    instruction = ep_data["instructions"][0]
                    episode = self._parse_episode(ep_data, instruction, str(i))
                    episodes.append(episode)
                elif self.instruction_variant == "random":
                    # Use random instruction (will be selected later)
                    import random
                    instruction = random.choice(ep_data["instructions"])
                    episode = self._parse_episode(ep_data, instruction, str(i))
                    episodes.append(episode)
                else:
                    raise ValueError(f"Unknown instruction variant: {self.instruction_variant}")

            except Exception as e:
                print(f"Error parsing episode {i}: {e}")
                continue

        self.episodes = episodes
        self._update_episode_dict()

        print(f"✅ Loaded {len(episodes)} R2R episodes")

    def _get_episode_file(self) -> Path:
        """Get path to episode file"""
        split = self.config.split

        if split not in R2R_SPLITS:
            raise ValueError(f"Unknown R2R split: {split}. Must be one of {R2R_SPLITS}")

        return self.data_path / f"R2R_{split}.json"

    def _download_dataset(self):
        """Download R2R dataset"""
        print("Downloading R2R dataset...")

        # Create directory
        self.data_path.mkdir(parents=True, exist_ok=True)

        # URLs for R2R data
        base_url = "https://github.com/peteanderson80/Matterport3DSimulator/raw/master/tasks/R2R/data"

        files = {
            "train": "R2R_train.json",
            "val_seen": "R2R_val_seen.json",
            "val_unseen": "R2R_val_unseen.json",
            "test": "R2R_test.json"
        }

        # Download current split file
        split = self.config.split
        if split in files:
            filename = files[split]
            url = f"{base_url}/{filename}"
            output_path = self.data_path / filename

            if not output_path.exists():
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, output_path)
                    print(f"✅ Downloaded {filename}")
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
            else:
                print(f"✅ {filename} already exists")

    def _parse_episode(self, ep_data: Dict[str, Any], instruction: str, episode_id: str) -> VLNEpisode:
        """Parse episode data into VLNEpisode"""
        # Extract basic info
        scene_id = ep_data["scan"]
        path_id = ep_data["path_id"]

        # Use path_id as base and add variant suffix if needed
        full_episode_id = f"{path_id}_{episode_id}" if "_" in episode_id else path_id

        # Parse start position and rotation
        start_pos = Position.from_list(ep_data["path"][0])
        start_rotation = [1, 0, 0, 0]  # Default quaternion
        if "heading" in ep_data:
            # Convert heading to quaternion (simplified)
            import math
            heading = ep_data["heading"]
            start_rotation = [
                math.cos(heading/2), 0, math.sin(heading/2), 0
            ]

        # Parse path
        path = [Position.from_list(pos) for pos in ep_data["path"]]

        # Create goal at end of path
        end_position = path[-1] if path else start_pos
        goal = Goal(
            goal_id="end_position",
            position=end_position,
            radius=3.0  # R2R success distance
        )

        # Additional info
        info = {
            "path_id": path_id,
            "distance": ep_data.get("distance", 0.0),
            "scan": scene_id,
            "heading": ep_data.get("heading", 0.0)
        }

        return VLNEpisode(
            episode_id=full_episode_id,
            scene_id=scene_id,
            start_position=start_pos,
            start_rotation=start_rotation,
            goals=[goal],
            info=info,
            instruction=instruction,
            path=path
        )

    def get_episode_count(self) -> int:
        """Get total episode count"""
        episode_file = self._get_episode_file()
        if not episode_file.exists():
            return 0

        try:
            with open(episode_file, 'r') as f:
                data = json.load(f)

            # Account for instruction variants
            if self.instruction_variant == "all":
                return sum(len(ep["instructions"]) for ep in data)
            else:
                return len(data)
        except:
            return 0

    def get_scene_ids(self) -> List[str]:
        """Get unique scene IDs"""
        if not self._loaded:
            self.load_episodes()
        return list(set(ep.scene_id for ep in self.episodes))

    def get_instruction_lengths(self) -> List[int]:
        """Get instruction lengths"""
        if not self._loaded:
            self.load_episodes()
        return [len(ep.instruction.split()) for ep in self.episodes]

    def get_path_lengths(self) -> List[float]:
        """Get path lengths"""
        if not self._loaded:
            self.load_episodes()
        return [ep.get_path_length() for ep in self.episodes]

    def filter_by_path_length(self, min_length: float = 0, max_length: float = float('inf')) -> 'R2RDataset':
        """Filter episodes by path length"""
        filtered_episodes = []
        for ep in self.episodes:
            path_length = ep.get_path_length()
            if min_length <= path_length <= max_length:
                filtered_episodes.append(ep)

        # Create filtered dataset
        new_config = DatasetConfig(
            name=f"{self.config.name}_length_{min_length}_{max_length}",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(filtered_episodes)
        )

        new_dataset = R2RDataset.__new__(R2RDataset)
        new_dataset.config = new_config
        new_dataset.data_path = self.data_path
        new_dataset.instruction_variant = self.instruction_variant
        new_dataset.episodes = filtered_episodes
        new_dataset._episode_dict = {ep.episode_id: ep for ep in filtered_episodes}
        new_dataset._loaded = True

        return new_dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed dataset statistics"""
        stats = super().get_statistics()

        if self.episodes:
            instruction_lengths = self.get_instruction_lengths()
            path_lengths = self.get_path_lengths()

            stats.update({
                "instruction_variant": self.instruction_variant,
                "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths),
                "avg_path_length": sum(path_lengths) / len(path_lengths),
                "min_path_length": min(path_lengths),
                "max_path_length": max(path_lengths),
                "total_instructions": len(self.episodes)
            })

        return stats


# Legacy compatibility
class R2RDatasetLoader(R2RDataset):
    """Legacy R2R dataset loader for compatibility"""

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = R2R_DATASET_DIR
        super().__init__(split="val_seen", data_path=data_dir)

    def load_split(self, split: str) -> List[Dict[str, Any]]:
        """Load episodes from a specific split"""
        dataset = R2RDataset(split=split, data_path=self.config.data_path)
        return [ep.to_dict() for ep in dataset.episodes]

    def download_r2r_data(self):
        """Download R2R dataset"""
        self._download_dataset()


def create_r2r_dataset(split: str = "val_seen",
                      max_episodes: Optional[int] = None,
                      instruction_variant: str = "all",
                      **kwargs) -> R2RDataset:
    """Create R2R dataset"""
    return R2RDataset(
        split=split,
        max_episodes=max_episodes,
        instruction_variant=instruction_variant,
        **kwargs
    )


def download_r2r_scenes():
    """Get R2R scene requirements"""
    print("Getting R2R scene list...")

    loader = R2RDatasetLoader()

    # Get all scenes used in R2R
    all_scenes = set()
    for split in R2R_SPLITS[:-1]:  # Exclude test split
        try:
            dataset = R2RDataset(split=split)
            scenes = dataset.get_scene_ids()
            all_scenes.update(scenes)
            print(f"{split}: {len(scenes)} scenes")
        except Exception as e:
            print(f"Failed to load {split}: {e}")

    print(f"\nTotal unique scenes in R2R: {len(all_scenes)}")
    print("Scene list:", sorted(list(all_scenes))[:10])  # Show first 10

    print(f"\nTo download these {len(all_scenes)} scenes:")
    print("python -m habitat_sim.utils.datasets_download --uids mp3d --data-path data")

    return sorted(list(all_scenes))


if __name__ == "__main__":
    # Demo usage
    print("=== R2R Dataset Demo ===")

    try:
        dataset = create_r2r_dataset("val_seen", max_episodes=5, instruction_variant="first")
        print(f"Dataset: {dataset.config.name}")
        print(f"Episodes: {len(dataset)}")

        if len(dataset) > 0:
            print("\nFirst episode:")
            ep = dataset[0]
            print(f"  ID: {ep.episode_id}")
            print(f"  Scene: {ep.scene_id}")
            print(f"  Instruction: {ep.instruction[:100]}...")
            print(f"  Path length: {ep.get_path_length():.1f}m")

        print("\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            if key != "scenes":
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("Downloading R2R dataset...")
        loader = R2RDatasetLoader()
        loader.download_r2r_data()