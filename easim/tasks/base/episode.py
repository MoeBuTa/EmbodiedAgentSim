"""
Base episode data structures for embodied AI tasks
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json


@dataclass
class Position:
    """3D position with optional rotation"""
    x: float
    y: float
    z: float
    rotation: Optional[List[float]] = None  # Quaternion [w, x, y, z]

    def to_list(self) -> List[float]:
        """Convert to list format"""
        return [self.x, self.y, self.z]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_list(cls, pos: List[float], rotation: Optional[List[float]] = None):
        """Create from list"""
        return cls(pos[0], pos[1], pos[2], rotation)


@dataclass
class Goal:
    """Base goal specification"""
    goal_id: str
    position: Optional[Position] = None
    description: Optional[str] = None
    object_category: Optional[str] = None
    object_id: Optional[str] = None
    radius: float = 0.5  # Success radius

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "goal_id": self.goal_id,
            "position": self.position.to_list() if self.position else None,
            "description": self.description,
            "object_category": self.object_category,
            "object_id": self.object_id,
            "radius": self.radius
        }


@dataclass
class BaseEpisode(ABC):
    """Base episode class for all tasks"""
    episode_id: str
    scene_id: str
    start_position: Position
    start_rotation: List[float]  # Quaternion [w, x, y, z]
    goals: List[Goal] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    shortest_paths: Optional[List[List[Position]]] = None

    def __post_init__(self):
        """Post-initialization processing"""
        self.validate()

    @abstractmethod
    def validate(self):
        """Validate episode data"""
        pass

    def get_start_position(self) -> List[float]:
        """Get start position as list"""
        return self.start_position.to_list()

    def get_start_rotation(self) -> List[float]:
        """Get start rotation"""
        return self.start_rotation

    def get_primary_goal(self) -> Optional[Goal]:
        """Get primary goal (first goal)"""
        return self.goals[0] if self.goals else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary"""
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "start_position": self.start_position.to_list(),
            "start_rotation": self.start_rotation,
            "goals": [goal.to_dict() for goal in self.goals],
            "info": self.info,
            "shortest_paths": [
                [pos.to_list() for pos in path]
                for path in (self.shortest_paths or [])
            ]
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEpisode':
        """Create episode from dictionary"""
        pass


@dataclass
class NavigationEpisode(BaseEpisode):
    """Episode for navigation tasks"""

    def validate(self):
        """Validate navigation episode"""
        if not self.goals:
            raise ValueError("Navigation episode must have at least one goal")

        # Check that goal has position
        primary_goal = self.get_primary_goal()
        if primary_goal and primary_goal.position is None:
            raise ValueError("Navigation goal must have a position")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NavigationEpisode':
        """Create navigation episode from dictionary"""
        # Parse positions
        start_pos = Position.from_list(data["start_position"])

        # Parse goals
        goals = []
        for goal_data in data.get("goals", []):
            goal = Goal(
                goal_id=goal_data["goal_id"],
                position=Position.from_list(goal_data["position"]) if goal_data.get("position") else None,
                description=goal_data.get("description"),
                object_category=goal_data.get("object_category"),
                object_id=goal_data.get("object_id"),
                radius=goal_data.get("radius", 0.5)
            )
            goals.append(goal)

        # Parse shortest paths
        shortest_paths = None
        if data.get("shortest_paths"):
            shortest_paths = [
                [Position.from_list(pos) for pos in path]
                for path in data["shortest_paths"]
            ]

        return cls(
            episode_id=data["episode_id"],
            scene_id=data["scene_id"],
            start_position=start_pos,
            start_rotation=data["start_rotation"],
            goals=goals,
            info=data.get("info", {}),
            shortest_paths=shortest_paths
        )


@dataclass
class PointNavEpisode(NavigationEpisode):
    """Episode for Point Navigation"""

    def validate(self):
        """Validate PointNav episode"""
        super().validate()

        # PointNav should have exactly one goal with position
        if len(self.goals) != 1:
            raise ValueError("PointNav episode must have exactly one goal")


@dataclass
class ObjectNavEpisode(NavigationEpisode):
    """Episode for Object Navigation"""

    def validate(self):
        """Validate ObjectNav episode"""
        super().validate()

        # ObjectNav goal must have object category
        primary_goal = self.get_primary_goal()
        if primary_goal and not primary_goal.object_category:
            raise ValueError("ObjectNav goal must specify object_category")


@dataclass
class VLNEpisode(NavigationEpisode):
    """Episode for Vision-Language Navigation"""
    instruction: str = ""
    path: List[Position] = field(default_factory=list)  # Ground truth path

    def validate(self):
        """Validate VLN episode"""
        if not self.instruction:
            raise ValueError("VLN episode must have instruction")

        if not self.path:
            raise ValueError("VLN episode must have ground truth path")

    def get_instruction(self) -> str:
        """Get navigation instruction"""
        return self.instruction

    def get_path_length(self) -> float:
        """Calculate path length"""
        if len(self.path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.path)):
            pos1 = self.path[i - 1].to_array()
            pos2 = self.path[i].to_array()
            total_length += np.linalg.norm(pos2 - pos1)

        return total_length

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "instruction": self.instruction,
            "path": [pos.to_list() for pos in self.path]
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VLNEpisode':
        """Create VLN episode from dictionary"""
        # Get base episode data
        base_episode = NavigationEpisode.from_dict(data)

        # Parse path
        path = [Position.from_list(pos) for pos in data.get("path", [])]

        return cls(
            episode_id=base_episode.episode_id,
            scene_id=base_episode.scene_id,
            start_position=base_episode.start_position,
            start_rotation=base_episode.start_rotation,
            goals=base_episode.goals,
            info=base_episode.info,
            shortest_paths=base_episode.shortest_paths,
            instruction=data["instruction"],
            path=path
        )


@dataclass
class EQAEpisode(BaseEpisode):
    """Episode for Embodied Question Answering"""
    question: str = ""
    answer: str = ""
    question_type: Optional[str] = None
    answer_type: Optional[str] = None  # e.g., "yes/no", "number", "color"

    def validate(self):
        """Validate EQA episode"""
        if not self.question:
            raise ValueError("EQA episode must have question")

        if not self.answer:
            raise ValueError("EQA episode must have answer")

    def get_question(self) -> str:
        """Get question text"""
        return self.question

    def get_answer(self) -> str:
        """Get answer text"""
        return self.answer

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = super().to_dict()
        data.update({
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "answer_type": self.answer_type
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EQAEpisode':
        """Create EQA episode from dictionary"""
        # Parse start position
        start_pos = Position.from_list(data["start_position"])

        # Parse goals (if any)
        goals = []
        for goal_data in data.get("goals", []):
            goal = Goal(
                goal_id=goal_data["goal_id"],
                position=Position.from_list(goal_data["position"]) if goal_data.get("position") else None,
                description=goal_data.get("description"),
                object_category=goal_data.get("object_category"),
                object_id=goal_data.get("object_id"),
                radius=goal_data.get("radius", 0.5)
            )
            goals.append(goal)

        return cls(
            episode_id=data["episode_id"],
            scene_id=data["scene_id"],
            start_position=start_pos,
            start_rotation=data["start_rotation"],
            goals=goals,
            info=data.get("info", {}),
            question=data["question"],
            answer=data["answer"],
            question_type=data.get("question_type"),
            answer_type=data.get("answer_type")
        )


class EpisodeIterator:
    """Iterator for episodes"""

    def __init__(self, episodes: List[BaseEpisode], shuffle: bool = False):
        self.episodes = episodes
        self.shuffle = shuffle
        self.current_index = 0

        if shuffle:
            import random
            random.shuffle(self.episodes)

    def __iter__(self):
        return self

    def __next__(self) -> BaseEpisode:
        if self.current_index >= len(self.episodes):
            raise StopIteration

        episode = self.episodes[self.current_index]
        self.current_index += 1
        return episode

    def __len__(self) -> int:
        return len(self.episodes)

    def reset(self):
        """Reset iterator"""
        self.current_index = 0

        if self.shuffle:
            import random
            random.shuffle(self.episodes)

    def get_current_episode(self) -> Optional[BaseEpisode]:
        """Get current episode without advancing"""
        if 0 <= self.current_index < len(self.episodes):
            return self.episodes[self.current_index]
        return None


class EpisodeDataset:
    """Dataset of episodes"""

    def __init__(self, episodes: List[BaseEpisode], name: str = ""):
        self.episodes = episodes
        self.name = name
        self._episode_dict = {ep.episode_id: ep for ep in episodes}

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: Union[int, str]) -> BaseEpisode:
        if isinstance(index, int):
            return self.episodes[index]
        elif isinstance(index, str):
            return self._episode_dict[index]
        else:
            raise TypeError("Index must be int or str")

    def get_episode_by_id(self, episode_id: str) -> Optional[BaseEpisode]:
        """Get episode by ID"""
        return self._episode_dict.get(episode_id)

    def get_episodes_by_scene(self, scene_id: str) -> List[BaseEpisode]:
        """Get episodes for specific scene"""
        return [ep for ep in self.episodes if ep.scene_id == scene_id]

    def get_unique_scenes(self) -> List[str]:
        """Get list of unique scene IDs"""
        return list(set(ep.scene_id for ep in self.episodes))

    def filter_by_scene(self, scene_ids: List[str]) -> 'EpisodeDataset':
        """Filter episodes by scene IDs"""
        filtered = [ep for ep in self.episodes if ep.scene_id in scene_ids]
        return EpisodeDataset(filtered, f"{self.name}_filtered")

    def sample(self, n: int, random_seed: Optional[int] = None) -> 'EpisodeDataset':
        """Sample n episodes"""
        import random
        if random_seed is not None:
            random.seed(random_seed)

        sampled = random.sample(self.episodes, min(n, len(self.episodes)))
        return EpisodeDataset(sampled, f"{self.name}_sampled_{n}")

    def shuffle(self, random_seed: Optional[int] = None) -> 'EpisodeDataset':
        """Shuffle episodes"""
        import random
        if random_seed is not None:
            random.seed(random_seed)

        shuffled = self.episodes.copy()
        random.shuffle(shuffled)
        return EpisodeDataset(shuffled, f"{self.name}_shuffled")

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
              random_seed: Optional[int] = None) -> Dict[str, 'EpisodeDataset']:
        """Split dataset into train/val/test"""
        import random
        if random_seed is not None:
            random.seed(random_seed)

        episodes = self.episodes.copy()
        random.shuffle(episodes)

        n_total = len(episodes)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_episodes = episodes[:n_train]
        val_episodes = episodes[n_train:n_train + n_val]
        test_episodes = episodes[n_train + n_val:]

        return {
            "train": EpisodeDataset(train_episodes, f"{self.name}_train"),
            "val": EpisodeDataset(val_episodes, f"{self.name}_val"),
            "test": EpisodeDataset(test_episodes, f"{self.name}_test")
        }

    def save_to_json(self, filepath: Union[str, Path]):
        """Save episodes to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "episodes": [ep.to_dict() for ep in self.episodes]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path],
                       episode_type: str = "navigation") -> 'EpisodeDataset':
        """Load episodes from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        episodes = []
        for ep_data in data["episodes"]:
            if episode_type == "navigation":
                episode = NavigationEpisode.from_dict(ep_data)
            elif episode_type == "pointnav":
                episode = PointNavEpisode.from_dict(ep_data)
            elif episode_type == "objectnav":
                episode = ObjectNavEpisode.from_dict(ep_data)
            elif episode_type == "vln":
                episode = VLNEpisode.from_dict(ep_data)
            elif episode_type == "eqa":
                episode = EQAEpisode.from_dict(ep_data)
            else:
                raise ValueError(f"Unknown episode type: {episode_type}")

            episodes.append(episode)

        return cls(episodes, data.get("name", ""))

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        scenes = self.get_unique_scenes()

        stats = {
            "num_episodes": len(self.episodes),
            "num_scenes": len(scenes),
            "scenes": scenes
        }

        # Type-specific statistics
        if self.episodes and isinstance(self.episodes[0], VLNEpisode):
            instructions = [ep.instruction for ep in self.episodes]
            avg_instruction_length = np.mean([len(inst.split()) for inst in instructions])
            path_lengths = [ep.get_path_length() for ep in self.episodes]

            stats.update({
                "avg_instruction_length": avg_instruction_length,
                "avg_path_length": np.mean(path_lengths),
                "min_path_length": np.min(path_lengths),
                "max_path_length": np.max(path_lengths)
            })

        elif self.episodes and isinstance(self.episodes[0], EQAEpisode):
            questions = [ep.question for ep in self.episodes]
            answers = [ep.answer for ep in self.episodes]

            stats.update({
                "avg_question_length": np.mean([len(q.split()) for q in questions]),
                "avg_answer_length": np.mean([len(a.split()) for a in answers]),
                "unique_answers": len(set(answers))
            })

        return stats


def create_episode_iterator(episodes: List[BaseEpisode],
                            shuffle: bool = False) -> EpisodeIterator:
    """Create episode iterator"""
    return EpisodeIterator(episodes, shuffle)


def create_episode_dataset(episodes: List[BaseEpisode],
                           name: str = "") -> EpisodeDataset:
    """Create episode dataset"""
    return EpisodeDataset(episodes, name)
