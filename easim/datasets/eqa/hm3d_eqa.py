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


def get_default_hm3d_eqa_config(split: str = "val") -> "DictConfig":
    """Get default config for HM3D EQA dataset"""
    return OmegaConf.create(  # type: ignore[call-overload]
        DatasetConfig(
            type="HM3DEQA-v1",
            split=split,
            data_path="data/datasets/eqa/hm3d/hm3d-eqa/hm3d-eqa.json",
        )
    )


@registry.register_dataset(name="HM3DEQA-v1")
class HM3DDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads HM3D
    Embodied Question Answering dataset.

    This class can then be used as follows::
        eqa_config.habitat.dataset = get_default_hm3d_eqa_config()
        eqa = habitat.make_task(eqa_config.habitat.task_name, config=eqa_config)
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
        """Load HM3D EQA dataset from JSON string"""
        deserialized = json.loads(json_str)
        
        for episode_data in deserialized["episodes"]:
            episode = self._process_episode(episode_data, scenes_dir)
            self.episodes.append(episode)
        
        # Build vocabularies from the loaded data
        self._build_vocabularies(deserialized)

    def _process_episode(
        self, episode_data: dict, scenes_dir: Optional[str] = None
    ) -> EQAEpisode:
        """Process a single episode from the dataset"""
        
        # Handle scene_id path resolution
        scene_id = episode_data["scene_id"]
        
        # HM3D EQA scene_ids need .glb extension added
        # Convert "hm3d_v0.2/train/00004-VqCaAuuoeWk" to "hm3d_v0.2/train/00004-VqCaAuuoeWk/VqCaAuuoeWk.basis.glb"
        if not scene_id.endswith('.glb'):
            scene_parts = scene_id.split('/')
            if len(scene_parts) >= 3:
                scene_name = scene_parts[-1].split('-')[-1]  # Extract "VqCaAuuoeWk" from "00004-VqCaAuuoeWk"
                scene_id = f"{scene_id}/{scene_name}.basis.glb"
        
        if scenes_dir is not None:
            if scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                scene_id = scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]
            scene_id = os.path.join(scenes_dir, scene_id)
        
        # Convert HM3D format to standard EQA format
        episode = EQAEpisode(
            episode_id=episode_data["episode_id"],
            scene_id=scene_id,
            start_position=episode_data["start_position"],
            start_rotation=episode_data["start_rotation"],
            question=QuestionData(**episode_data["question"]),
            goals=[],  # HM3D EQA doesn't have goals
            shortest_paths=None,  # HM3D EQA doesn't have shortest paths
        )
        
        # Add HM3D-specific fields as extra info
        if hasattr(episode, 'info'):
            episode.info = episode.info or {}
        else:
            episode.info = {}
            
        episode.info.update({
            'scene': episode_data.get('scene'),
            'floor': episode_data.get('floor'),
            'choices': episode_data.get('choices', []),
            'question_formatted': episode_data.get('question_formatted'),
            'label': episode_data.get('label'),
        })
        
        return episode

    def _build_vocabularies(self, data: dict) -> None:
        """Build question and answer vocabularies from the dataset"""
        # Collect all unique question tokens and answer texts
        question_tokens = set()
        answer_texts = set()
        
        for episode_data in data["episodes"]:
            question = episode_data.get("question", {})
            
            # Add question tokens
            if "question_tokens" in question:
                question_tokens.update(question["question_tokens"])
            
            # Add answer text
            if "answer_text" in question:
                answer_texts.add(question["answer_text"])
            
            # Also add choices as potential answers
            choices = episode_data.get("choices", [])
            answer_texts.update(choices)
        
        # Build vocabularies
        self.question_vocab = VocabDict(word_list=sorted(list(question_tokens)))
        self.answer_vocab = VocabDict(word_list=sorted(list(answer_texts)))