"""
Base dataset classes for embodied AI tasks
"""
import json
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass

from easim.tasks.base.episode import BaseEpisode, EpisodeDataset, EpisodeIterator


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    split: str = "val"
    data_path: Optional[str] = None
    max_episodes: Optional[int] = None
    shuffle: bool = False
    cache_episodes: bool = True
    scene_filter: Optional[List[str]] = None


class BaseDataset(ABC):
    """Abstract base class for embodied AI datasets"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.episodes: List[BaseEpisode] = []
        self._episode_dict: Dict[str, BaseEpisode] = {}
        self._loaded = False

        # Load episodes if not lazy loading
        if not getattr(self, '_lazy_load', False):
            self.load_episodes()

    @abstractmethod
    def load_episodes(self):
        """Load episodes from dataset"""
        pass

    @abstractmethod
    def get_episode_count(self) -> int:
        """Get total number of episodes in dataset"""
        pass

    @abstractmethod
    def get_scene_ids(self) -> List[str]:
        """Get list of scene IDs in dataset"""
        pass

    def __len__(self) -> int:
        """Get number of loaded episodes"""
        return len(self.episodes)

    def __getitem__(self, index: Union[int, str]) -> BaseEpisode:
        """Get episode by index or ID"""
        if isinstance(index, int):
            return self.episodes[index]
        elif isinstance(index, str):
            return self._episode_dict[index]
        else:
            raise TypeError("Index must be int or str")

    def __iter__(self) -> Iterator[BaseEpisode]:
        """Iterate over episodes"""
        return iter(self.episodes)

    def get_episode_by_id(self, episode_id: str) -> Optional[BaseEpisode]:
        """Get episode by ID"""
        return self._episode_dict.get(episode_id)

    def get_episodes_by_scene(self, scene_id: str) -> List[BaseEpisode]:
        """Get episodes for specific scene"""
        return [ep for ep in self.episodes if ep.scene_id == scene_id]

    def filter_by_scenes(self, scene_ids: List[str]) -> 'BaseDataset':
        """Filter dataset by scene IDs"""
        filtered_episodes = [ep for ep in self.episodes if ep.scene_id in scene_ids]

        # Create new dataset with filtered episodes
        new_config = DatasetConfig(
            name=f"{self.config.name}_filtered",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(filtered_episodes),
            shuffle=self.config.shuffle,
            cache_episodes=self.config.cache_episodes
        )

        new_dataset = self.__class__(new_config)
        new_dataset.episodes = filtered_episodes
        new_dataset._episode_dict = {ep.episode_id: ep for ep in filtered_episodes}
        new_dataset._loaded = True

        return new_dataset

    def sample(self, n: int, random_seed: Optional[int] = None) -> 'BaseDataset':
        """Sample n episodes from dataset"""
        if random_seed is not None:
            np.random.seed(random_seed)

        sampled_episodes = np.random.choice(
            self.episodes,
            size=min(n, len(self.episodes)),
            replace=False
        ).tolist()

        # Create new dataset with sampled episodes
        new_config = DatasetConfig(
            name=f"{self.config.name}_sampled_{n}",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(sampled_episodes),
            shuffle=self.config.shuffle,
            cache_episodes=self.config.cache_episodes
        )

        new_dataset = self.__class__(new_config)
        new_dataset.episodes = sampled_episodes
        new_dataset._episode_dict = {ep.episode_id: ep for ep in sampled_episodes}
        new_dataset._loaded = True

        return new_dataset

    def shuffle(self, random_seed: Optional[int] = None) -> 'BaseDataset':
        """Shuffle episodes"""
        if random_seed is not None:
            np.random.seed(random_seed)

        shuffled_episodes = self.episodes.copy()
        np.random.shuffle(shuffled_episodes)

        # Create new dataset with shuffled episodes
        new_config = DatasetConfig(
            name=f"{self.config.name}_shuffled",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(shuffled_episodes),
            shuffle=True,
            cache_episodes=self.config.cache_episodes
        )

        new_dataset = self.__class__(new_config)
        new_dataset.episodes = shuffled_episodes
        new_dataset._episode_dict = {ep.episode_id: ep for ep in shuffled_episodes}
        new_dataset._loaded = True

        return new_dataset

    def split_dataset(self,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      random_seed: Optional[int] = None) -> Dict[str, 'BaseDataset']:
        """Split dataset into train/val/test"""
        if random_seed is not None:
            np.random.seed(random_seed)

        episodes = self.episodes.copy()
        np.random.shuffle(episodes)

        n_total = len(episodes)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_episodes = episodes[:n_train]
        val_episodes = episodes[n_train:n_train + n_val]
        test_episodes = episodes[n_train + n_val:]

        splits = {}
        for split_name, split_episodes in [
            ("train", train_episodes),
            ("val", val_episodes),
            ("test", test_episodes)
        ]:
            config = DatasetConfig(
                name=f"{self.config.name}_{split_name}",
                split=split_name,
                data_path=self.config.data_path,
                max_episodes=len(split_episodes),
                shuffle=self.config.shuffle,
                cache_episodes=self.config.cache_episodes
            )

            dataset = self.__class__(config)
            dataset.episodes = split_episodes
            dataset._episode_dict = {ep.episode_id: ep for ep in split_episodes}
            dataset._loaded = True

            splits[split_name] = dataset

        return splits

    def to_episode_dataset(self) -> EpisodeDataset:
        """Convert to EpisodeDataset"""
        return EpisodeDataset(self.episodes, self.config.name)

    def save_to_json(self, filepath: Union[str, Path]):
        """Save dataset to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "name": self.config.name,
                "split": self.config.split,
                "max_episodes": self.config.max_episodes
            },
            "episodes": [ep.to_dict() for ep in self.episodes]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path], episode_type: str) -> 'BaseDataset':
        """Load dataset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create config
        config_data = data.get("config", {})
        config = DatasetConfig(**config_data)

        # Create dataset
        dataset = cls(config)

        # Load episodes
        from ...tasks.base.episode import (
            NavigationEpisode, PointNavEpisode, ObjectNavEpisode,
            VLNEpisode, EQAEpisode
        )

        episode_classes = {
            "navigation": NavigationEpisode,
            "pointnav": PointNavEpisode,
            "objectnav": ObjectNavEpisode,
            "vln": VLNEpisode,
            "eqa": EQAEpisode
        }

        if episode_type not in episode_classes:
            raise ValueError(f"Unknown episode type: {episode_type}")

        episode_class = episode_classes[episode_type]
        episodes = []

        for ep_data in data["episodes"]:
            episode = episode_class.from_dict(ep_data)
            episodes.append(episode)

        dataset.episodes = episodes
        dataset._episode_dict = {ep.episode_id: ep for ep in episodes}
        dataset._loaded = True

        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.episodes:
            return {"num_episodes": 0}

        scene_ids = list(set(ep.scene_id for ep in self.episodes))

        stats = {
            "name": self.config.name,
            "split": self.config.split,
            "num_episodes": len(self.episodes),
            "num_scenes": len(scene_ids),
            "scenes": sorted(scene_ids)
        }

        return stats

    def _update_episode_dict(self):
        """Update episode dictionary after loading"""
        self._episode_dict = {ep.episode_id: ep for ep in self.episodes}
        self._loaded = True


class FileBasedDataset(BaseDataset):
    """Base class for file-based datasets"""

    def __init__(self, config: DatasetConfig, data_files: List[Path]):
        self.data_files = data_files
        super().__init__(config)

    def get_episode_count(self) -> int:
        """Get episode count from files"""
        # Override in subclasses for efficient counting
        if not self._loaded:
            self.load_episodes()
        return len(self.episodes)

    def get_scene_ids(self) -> List[str]:
        """Get scene IDs from episodes"""
        if not self._loaded:
            self.load_episodes()
        return list(set(ep.scene_id for ep in self.episodes))


class HabitatDataset(BaseDataset):
    """Base class for Habitat-compatible datasets"""

    def __init__(self, config: DatasetConfig, habitat_config_path: Optional[Path] = None):
        self.habitat_config_path = habitat_config_path
        super().__init__(config)

    def get_habitat_config(self):
        """Get Habitat configuration if available"""
        if self.habitat_config_path and self.habitat_config_path.exists():
            try:
                import habitat
                return habitat.get_config(str(self.habitat_config_path))
            except ImportError:
                return None
        return None


def create_dataset_config(name: str,
                          split: str = "val",
                          max_episodes: Optional[int] = None,
                          **kwargs) -> DatasetConfig:
    """Create dataset configuration"""
    return DatasetConfig(
        name=name,
        split=split,
        max_episodes=max_episodes,
        **kwargs
    )