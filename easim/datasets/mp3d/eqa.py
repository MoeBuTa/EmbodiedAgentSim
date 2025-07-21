"""
MP3D Embodied Question Answering (EQA) dataset
"""
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional

from easim.datasets.base.dataset import FileBasedDataset, DatasetConfig
from easim.tasks.base.episode import EQAEpisode, Position, Goal
from easim.utils.constants import MP3D_EQA_DIR


class MP3DEQADataset(FileBasedDataset):
    """MP3D Embodied Question Answering dataset"""

    def __init__(self,
                 split: str = "val",
                 max_episodes: Optional[int] = None,
                 data_path: Optional[str] = None,
                 question_type_filter: Optional[str] = None,
                 **kwargs):

        # Setup paths
        if data_path is None:
            data_path = MP3D_EQA_DIR

        config = DatasetConfig(
            name=f"mp3d_eqa_{split}",
            split=split,
            data_path=str(data_path),
            max_episodes=max_episodes,
            **kwargs
        )

        self.data_path = Path(data_path)
        self.question_type_filter = question_type_filter

        # Get data files
        data_files = [self._get_episode_file()]

        super().__init__(config, data_files)

    def load_episodes(self):
        """Load MP3D EQA episodes"""
        episode_file = self._get_episode_file()

        if not episode_file.exists():
            self._download_dataset()

        if not episode_file.exists():
            print(f"Episode file not found: {episode_file}")
            print("Please download MP3D EQA dataset manually")
            return

        print(f"Loading MP3D EQA episodes from {episode_file}")

        # Load episodes from file
        if episode_file.suffix == '.gz':
            with gzip.open(episode_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(episode_file, 'r') as f:
                data = json.load(f)

        episodes = []
        episode_list = data.get('episodes', data)  # Handle different formats

        # Apply filters
        if self.question_type_filter:
            episode_list = [ep for ep in episode_list
                            if self._get_question_type(ep.get('question', '')) == self.question_type_filter]

        # Apply max_episodes limit
        if self.config.max_episodes:
            episode_list = episode_list[:self.config.max_episodes]

        for i, ep_data in enumerate(episode_list):
            try:
                episode = self._parse_episode(ep_data, i)
                episodes.append(episode)
            except Exception as e:
                print(f"Error parsing episode {i}: {e}")
                continue

        self.episodes = episodes
        self._update_episode_dict()

        print(f"✅ Loaded {len(episodes)} MP3D EQA episodes")

    def _get_episode_file(self) -> Path:
        """Get path to episode file"""
        split = self.config.split

        # Common MP3D EQA file patterns
        possible_files = [
            self.data_path / f"mp3d_eqa_v1_{split}.json.gz",
            self.data_path / f"eqa_mp3d_{split}.json.gz",
            self.data_path / f"{split}.json.gz",
            self.data_path / f"mp3d_eqa_{split}.json",
            self.data_path / f"{split}.json"
        ]

        for file_path in possible_files:
            if file_path.exists():
                return file_path

        # Return the most likely path for download
        return self.data_path / f"mp3d_eqa_v1_{split}.json.gz"

    def _download_dataset(self):
        """Download MP3D EQA dataset"""
        print("MP3D EQA dataset not found")
        print("Please download from: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md")
        print(f"Expected location: {self._get_episode_file()}")

        # Create directory
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _parse_episode(self, ep_data: Dict[str, Any], index: int) -> EQAEpisode:
        """Parse episode data into EQAEpisode"""
        # Extract basic info
        episode_id = ep_data.get("episode_id", f"eqa_{index}")
        scene_id = ep_data.get("scene_id", ep_data.get("scan", ""))

        # Extract question and answer
        question = ep_data.get("question", "")
        answer = ep_data.get("answer", "")

        # Parse start position and rotation
        start_position = ep_data.get("start_position", [0, 0, 0])
        start_pos = Position.from_list(start_position)

        start_rotation = ep_data.get("start_rotation", [1, 0, 0, 0])

        # Determine question and answer types
        question_type = self._get_question_type(question)
        answer_type = self._get_answer_type(answer)

        # Parse goals if any (EQA might not have explicit spatial goals)
        goals = []
        if "goals" in ep_data:
            for goal_data in ep_data["goals"]:
                position = None
                if "position" in goal_data:
                    position = Position.from_list(goal_data["position"])

                goal = Goal(
                    goal_id=goal_data.get("goal_id", "0"),
                    position=position,
                    description=goal_data.get("description"),
                    radius=goal_data.get("radius", 0.5)
                )
                goals.append(goal)

        # Additional info
        info = {
            "question_length": len(question.split()),
            "answer_length": len(answer.split()),
            "difficulty": ep_data.get("difficulty", "medium")
        }

        return EQAEpisode(
            episode_id=episode_id,
            scene_id=scene_id,
            start_position=start_pos,
            start_rotation=start_rotation,
            goals=goals,
            info=info,
            question=question,
            answer=answer,
            question_type=question_type,
            answer_type=answer_type
        )

    def _get_question_type(self, question: str) -> str:
        """Determine question type from question text"""
        question_lower = question.lower()

        # Simple heuristics for question type classification
        if any(word in question_lower for word in ["how many", "count"]):
            return "counting"
        elif any(word in question_lower for word in ["what color", "color"]):
            return "color"
        elif any(word in question_lower for word in ["where", "location"]):
            return "spatial"
        elif any(word in question_lower for word in ["is there", "are there", "exists"]):
            return "existence"
        elif any(word in question_lower for word in ["what room", "room"]):
            return "room_type"
        elif any(word in question_lower for word in ["what", "which"]):
            return "object_category"
        else:
            return "other"

    def _get_answer_type(self, answer: str) -> str:
        """Determine answer type from answer text"""
        answer_lower = answer.lower().strip()

        # Simple heuristics for answer type classification
        if answer_lower in ["yes", "no"]:
            return "yes/no"
        elif answer_lower.isdigit():
            return "number"
        elif any(color in answer_lower for color in ["red", "blue", "green", "yellow", "black", "white", "brown"]):
            return "color"
        elif any(room in answer_lower for room in ["kitchen", "bedroom", "bathroom", "living room", "dining room"]):
            return "room"
        else:
            return "object"

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

            episode_list = data.get('episodes', data)
            return len(episode_list)
        except:
            return 0

    def get_scene_ids(self) -> List[str]:
        """Get unique scene IDs"""
        if not self._loaded:
            self.load_episodes()
        return list(set(ep.scene_id for ep in self.episodes))

    def get_question_types(self) -> List[str]:
        """Get unique question types"""
        if not self._loaded:
            self.load_episodes()
        return list(set(ep.question_type for ep in self.episodes if ep.question_type))

    def get_answer_types(self) -> List[str]:
        """Get unique answer types"""
        if not self._loaded:
            self.load_episodes()
        return list(set(ep.answer_type for ep in self.episodes if ep.answer_type))

    def filter_by_question_type(self, question_type: str) -> 'MP3DEQADataset':
        """Filter episodes by question type"""
        filtered_episodes = []
        for ep in self.episodes:
            if ep.question_type == question_type:
                filtered_episodes.append(ep)

        # Create filtered dataset
        new_config = DatasetConfig(
            name=f"{self.config.name}_{question_type}",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(filtered_episodes)
        )

        new_dataset = MP3DEQADataset.__new__(MP3DEQADataset)
        new_dataset.config = new_config
        new_dataset.data_path = self.data_path
        new_dataset.question_type_filter = question_type
        new_dataset.episodes = filtered_episodes
        new_dataset._episode_dict = {ep.episode_id: ep for ep in filtered_episodes}
        new_dataset._loaded = True

        return new_dataset

    def filter_by_answer_type(self, answer_type: str) -> 'MP3DEQADataset':
        """Filter episodes by answer type"""
        filtered_episodes = []
        for ep in self.episodes:
            if ep.answer_type == answer_type:
                filtered_episodes.append(ep)

        # Create filtered dataset
        new_config = DatasetConfig(
            name=f"{self.config.name}_{answer_type}",
            split=self.config.split,
            data_path=self.config.data_path,
            max_episodes=len(filtered_episodes)
        )

        new_dataset = MP3DEQADataset.__new__(MP3DEQADataset)
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
            question_types = {}
            answer_types = {}
            question_lengths = []
            answer_lengths = []

            for ep in self.episodes:
                # Count question types
                if ep.question_type:
                    question_types[ep.question_type] = question_types.get(ep.question_type, 0) + 1

                # Count answer types
                if ep.answer_type:
                    answer_types[ep.answer_type] = answer_types.get(ep.answer_type, 0) + 1

                # Collect lengths
                question_lengths.append(len(ep.question.split()))
                answer_lengths.append(len(ep.answer.split()))

            stats.update({
                "question_types": len(question_types),
                "answer_types": len(answer_types),
                "question_type_distribution": question_types,
                "answer_type_distribution": answer_types,
                "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
                "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
            })

        return stats


def create_mp3d_eqa_dataset(split: str = "val",
                            max_episodes: Optional[int] = None,
                            question_type: Optional[str] = None,
                            answer_type: Optional[str] = None,
                            **kwargs) -> MP3DEQADataset:
    """Create MP3D EQA dataset"""
    dataset = MP3DEQADataset(
        split=split,
        max_episodes=max_episodes,
        question_type_filter=question_type,
        **kwargs
    )

    if answer_type:
        dataset = dataset.filter_by_answer_type(answer_type)

    return dataset


def download_mp3d_eqa():
    """Download MP3D EQA dataset"""
    print("To download MP3D EQA dataset:")
    print("1. Visit: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md")
    print("2. Download EQA episodes for MP3D")
    print("3. Extract to:", MP3D_EQA_DIR)
    print()
    print("Expected files:")
    for split in ["train", "val"]:
        print(f"  - mp3d_eqa_v1_{split}.json.gz")


if __name__ == "__main__":
    # Demo usage
    print("=== MP3D EQA Dataset Demo ===")

    try:
        dataset = create_mp3d_eqa_dataset("val", max_episodes=5)
        print(f"Dataset: {dataset.config.name}")
        print(f"Episodes: {len(dataset)}")

        if len(dataset) > 0:
            print("\nFirst episode:")
            ep = dataset[0]
            print(f"  ID: {ep.episode_id}")
            print(f"  Scene: {ep.scene_id}")
            print(f"  Question: {ep.question}")
            print(f"  Answer: {ep.answer}")
            print(f"  Question Type: {ep.question_type}")
            print(f"  Answer Type: {ep.answer_type}")

        print("\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            if key not in ["scenes", "question_type_distribution", "answer_type_distribution"]:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Demo failed: {e}")
        download_mp3d_eqa()