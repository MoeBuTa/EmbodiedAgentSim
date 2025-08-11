import json
import os
from typing import TYPE_CHECKING, List, Optional

from omegaconf import OmegaConf

from habitat.config.default_structured_configs import DatasetConfig
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.tasks.eqa.eqa import EQAEpisode, QuestionData

if TYPE_CHECKING:
    from habitat.config import DictConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


def get_default_hm3d_express_config(split: str = "val") -> "DictConfig":
    """Get default config for HM3D Express dataset"""
    return OmegaConf.create(  # type: ignore[call-overload]
        DatasetConfig(
            type="HM3DExpress-v1",
            split=split,
            data_path="data/datasets/eqa/hm3d/express-bench/express-bench.json",
        )
    )


@registry.register_dataset(name="HM3DExpress-v1")
class HM3DExpressDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads HM3D
    Express Embodied Question Answering dataset.

    This class can then be used as follows::
        express_config.habitat.dataset = get_default_hm3d_express_config()
        express = habitat.make_task(express_config.habitat.task_name, config=express_config)
    """

    episodes: List[EQAEpisode]
    answer_vocab: VocabDict
    question_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(config.data_path)

    def __init__(self, config: "DictConfig" = None) -> None:
        self.episodes = []

        if config is None:
            return

        with open(config.data_path, "r") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        """Load HM3D Express dataset from JSON string"""
        deserialized = json.loads(json_str)
        
        # Express-bench format is a list of episodes directly
        for episode_data in deserialized:
            episode = self._process_episode(episode_data, scenes_dir)
            self.episodes.append(episode)
        
        # Build vocabularies from the loaded data
        self._build_vocabularies(deserialized)

    def _process_episode(
        self, episode_data: dict, scenes_dir: Optional[str] = None
    ) -> EQAEpisode:
        """Process a single episode from the Express dataset"""
        
        # Handle scene_id path resolution
        scene_id = episode_data["scene_id"]
        
        # Express scene_ids need .glb extension added
        # Convert "hm3d_v0.2/train/00006-HkseAnWCgqk" to "hm3d_v0.2/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb"
        if not scene_id.endswith('.glb'):
            scene_parts = scene_id.split('/')
            if len(scene_parts) >= 3:
                scene_name = scene_parts[-1].split('-')[-1]  # Extract "HkseAnWCgqk" from "00006-HkseAnWCgqk"
                scene_id = f"{scene_id}/{scene_name}.basis.glb"
        
        if scenes_dir is not None:
            if scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                scene_id = scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]
            scene_id = os.path.join(scenes_dir, scene_id)
        
        # Create QuestionData from Express format
        question_data = QuestionData(
            question_text=episode_data["question"],
            answer_text=episode_data["answer"],
            question_tokens=episode_data["question"].split(),  # Simple tokenization
        )
        
        # Convert Express format to standard EQA format
        episode = EQAEpisode(
            episode_id=str(episode_data["episode_id"]),
            scene_id=scene_id,
            start_position=episode_data["start_position"],
            start_rotation=episode_data["start_rotation"],
            question=question_data,
            goals=[],  # Express doesn't have explicit goals
            shortest_paths=None,  # Express doesn't have shortest paths
        )
        
        # Add Express-specific fields as extra info
        if hasattr(episode, 'info'):
            episode.info = episode.info or {}
        else:
            episode.info = {}
            
        episode.info.update({
            'trajectory_id': episode_data.get('trajectory_id'),
            'episode_path': episode_data.get('episode_path'),
            'type': episode_data.get('type'),
            'goal_position': episode_data.get('goal_position'),
            'goal_rotation': episode_data.get('goal_rotation'),
            'geodesic_distance': episode_data.get('geodesic_distance'),
            'step_length': episode_data.get('step_length'),
            'actions': episode_data.get('actions'),
        })
        
        return episode

    def _build_vocabularies(self, data: List[dict]) -> None:
        """Build question and answer vocabularies from the dataset"""
        # Collect all unique question tokens and answer texts
        question_tokens = set()
        answer_texts = set()
        
        for episode_data in data:
            # Add question tokens (simple word tokenization)
            question_text = episode_data.get("question", "")
            question_tokens.update(question_text.split())
            
            # Add answer text
            answer_text = episode_data.get("answer", "")
            if answer_text:
                answer_texts.add(answer_text)
        
        # Build vocabularies
        self.question_vocab = VocabDict(word_list=sorted(list(question_tokens)))
        self.answer_vocab = VocabDict(word_list=sorted(list(answer_texts)))