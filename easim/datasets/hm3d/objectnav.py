"""
MP3D Object Navigation dataset
"""
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional

from easim.datasets.base.dataset import HabitatDataset, DatasetConfig
from easim.tasks.base.episode import ObjectNavEpisode, Position, Goal
from easim.utils.constants import (
    MP3D_OBJECTNAV_DIR, MP3D_CONFIG_PATH, MP3D_OBJECTNAV_CATEGORIES
)


class MP3DObjectNavDataset(HabitatDataset):
    """MP3D Object Navigation dataset"""

    def __init__(self,
                 split: str = "val",
                 max_episodes: Optional[int] = None,
                 data_path: Optional[str] = None,
                 **kwargs):

        # Setup paths
        if data_path is None:
            data_path = MP3D_OBJECTNAV_DIR

        config = DatasetConfig(
            name=f"mp3d_objectnav_{split}",
            split=split,
            data_path=str(data_path),
            max_episodes=max_episodes,
            **kwargs
        )

        self.data_path = Path(data_path)
        super().__init__(config, MP3D_CONFIG_PATH)

    def load_episodes(self):
        """Load MP3D ObjectNav episodes"""
        episode_file = self._get_episode_file()

        if not episode_file.exists():
            self._download_dataset()

        if not episode_file.exists():
            print(f"Episode file not found: {episode_file}")
            print("Please download MP3D ObjectNav dataset manually")
            return

        print(f"Loading MP3D ObjectNav episodes from {episode_file}")

        # Load episodes from file
        if episode_file.suffix == '.gz':
            with gzip.open(episode_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(episode_file, 'r') as f:
                data = json.load(f)

        episodes = []
        episode_list = data.get('episodes', [])

        # Apply max_episodes limit
        if self.config.max_episodes:
            episode_list = episode_list[:self.config.max_episodes]

        for i, ep_data in enumerate(episode_list):
            try:
                episode = self._parse_episode(ep_data)
                episodes.append(episode)
            except Exception as e:
                print(f"Error parsing episode {i}: {e}")
                continue

        self.episodes = episodes
        self._update_episode_dict()

        print(f"✅ Loaded {len(episodes)} MP3D ObjectNav episodes")

    def _get_episode_file(self) -> Path:
        """Get path to episode file"""
        split = self.config.split

        # Common MP3D ObjectNav file patterns
        possible_files = [
            self.data_path / f"mp3d_v1_{split}.json.gz",
            self.data_path / f"objectnav_mp3d_v1_{split}.json.gz",
            self.data_path / f"{split}.json.gz",
            self.data_path / f"mp3d_objectnav_{split}.json",
            self.data_path / f"{split}.json"
        ]

        for file_path in possible_files:
            if file_path.exists():
                return file_path

        # Return the most likely path for download
        return self.data_path / f"mp3d_v1_{split}.json.gz"

    def _download_dataset(self):
        """Download MP3D ObjectNav dataset"""
        print("MP3D ObjectNav dataset not found")
        print("Please download from: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md")
        print(f"Expected location: {self._get_episode_file()}")

        # Create directory
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _parse_episode(self, ep_data: Dict[str, Any]) -> ObjectNavEpisode:
        """Parse episode data into ObjectNavEpisode"""
        # Extract basic info
        episode_id = ep_data["episode_id"]
        scene_id = ep_data["scene_id"]

        # Parse start position and rotation
        start_pos = Position.from_list(ep_data["start_position"])
        start_rotation = ep_data["start_rotation"]

        # Parse goals
        goals = []
        for goal_data in ep_data.get("goals", []):
            # ObjectNav goals contain object category
            object_category = goal_data.get("object_category")
            position = None

            if "position" in goal_data:
                position = Position.from_list(goal_data["position"])

            goal = Goal(
                goal_id=goal_data.get("goal_id", "0"),
                position=position,
                object_category=object_category,
                radius=goal_data.get("radius", 1.0)
            )
            goals.append(goal)

        # Parse shortest paths if available
        shortest_paths = None
        if "shortest_paths" in ep_data:
            shortest_paths = []
            for path_data in ep_data["shortest_paths"]:
                path = [Position.from_list(pos) for pos in path_data]
                shortest_paths.append(path)

        # Additional info
        info = {
            "geodesic_distance": ep_data.get("info", {}).get("geodesic_distance", 0.0),
            "euclidean_distance": ep_data.get("info", {}).get("euclidean_distance", 0.0)
        }

        return ObjectNavEpisode(
            episode_id=episode_id,
            scene_id=scene_id,
            start_position=start_pos,
            start_rotation=start_rotation,
            goals=goals,
            info=info,
            shortest_paths=shortest_paths
        )

    def get_episode_count(self) -> int:
        """Get total episode count"""
        episode_file = self._get_episode_file()
        if not episode_file.exists():
            return 0

        try:
            if episode_file.suffix == '.gz':
                with gzip.open(episode_file, 'rt') as f:
                    data = json.load(f)
            else:
                with open(episode_file, 'r') as f:
                    data = json.load(f)

            return len(data.get('episodes', []))
        except:
            return 0

    def get_scene_ids(self) -> List[str]:
        """Get unique scene IDs"""
        if not self._loaded:
            self.load_episodes()
        return list(set(ep.scene_id for ep in self.episodes))

    def get_object_categories(self) -> List[str]:
        """Get object categories used in dataset"""
        return MP3D_OBJECTNAV_CATEGORIES

    def filter_by_object_category(self, category: str) -> 'MP3DObjectNavDataset':
        """Filter episodes by object category"""
        filtered_episodes = []
        for ep in self.episodes:
            if ep.goals and ep.goals[0].object_category == category:
                filtered_episodes.append(ep)

        # Create filtered dataset
        new_config = DatasetConfig(
            name=f"{self.config.name}_{category}",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(filtered_episodes)
        )

        new_dataset = MP3DObjectNavDataset.__new__(MP3DObjectNavDataset)
        new_dataset.config = new_config
        new_dataset.data_path = self.data_path
        new_dataset.episodes = filtered_episodes
        new_dataset._episode_dict = {ep.episode_id: ep for ep in filtered_episodes}
        new_dataset._loaded = True

        return new_dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed dataset statistics"""
        stats = super().get_statistics()

        if self.episodes:
            # Object category distribution
            category_counts = {}
            for ep in self.episodes:
                if ep.goals and ep.goals[0].object_category:
                    cat = ep.goals[0].object_category
                    category_counts[cat] = category_counts.get(cat, 0) + 1

            stats.update({
                "object_categories": len(category_counts),
                "category_distribution": category_counts,
                "avg_geodesic_distance": sum(
                    ep.info.get("geodesic_distance", 0) for ep in self.episodes
                ) / len(self.episodes) if self.episodes else 0
            })

        return stats


def create_mp3d_objectnav_dataset(split: str = "val",
                                  max_episodes: Optional[int] = None,
                                  object_category: Optional[str] = None,
                                  **kwargs) -> MP3DObjectNavDataset:
    """Create MP3D ObjectNav dataset"""
    dataset = MP3DObjectNavDataset(
        split=split,
        max_episodes=max_episodes,
        **kwargs
    )

    if object_category:
        dataset = dataset.filter_by_object_category(object_category)

    return dataset


def download_mp3d_objectnav():
    """Download MP3D ObjectNav dataset"""
    print("To download MP3D ObjectNav dataset:")
    print("1. Visit: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md")
    print("2. Download ObjectNav episodes for MP3D")
    print("3. Extract to:", MP3D_OBJECTNAV_DIR)
    print()
    print("Expected files:")
    for split in ["train", "val"]:
        print(f"  - mp3d_v1_{split}.json.gz")


if __name__ == "__main__":
    # Demo usage
    print("=== MP3D ObjectNav Dataset Demo ===")

    try:
        dataset = create_mp3d_objectnav_dataset("val", max_episodes=5)
        print(f"Dataset: {dataset.config.name}")
        print(f"Episodes: {len(dataset)}")

        if len(dataset) > 0:
            print("\nFirst episode:")
            ep = dataset[0]
            print(f"  ID: {ep.episode_id}")
            print(f"  Scene: {ep.scene_id}")
            print(f"  Target: {ep.goals[0].object_category if ep.goals else 'None'}")

        print("\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            if key != "scenes":
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Demo failed: {e}")
        download_mp3d_objectnav()