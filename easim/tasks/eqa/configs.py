"""
MP3D EQA configuration dataclasses
"""
from dataclasses import dataclass, field
from typing import List, Optional

from easim.tasks.base.configs import TaskConfig


@dataclass
class EQAConfig(TaskConfig):
    """EQA specific configuration"""
    # Answer vocabulary
    answer_vocab_size: int = 3000
    answer_vocab_file: Optional[str] = None

    # Question types
    question_types: List[str] = field(default_factory=lambda: [
        "existence", "counting", "spatial", "color", "object_category",
        "room_type", "relative_position", "size", "material"
    ])

    # Navigation vs pure QA
    allow_navigation: bool = True
    max_navigation_steps: int = 100
    answer_after_navigation: bool = True

    # Rewards
    correct_answer_reward: float = 10.0
    wrong_answer_penalty: float = -1.0
    navigation_step_penalty: float = -0.01

    def __post_init__(self):
        super().__post_init__()
        self.action_space_type = "eqa"
        self.sensor_suite_type = "eqa"
        self.max_episode_steps = 200  # EQA episodes are typically shorter


@dataclass
class MP3DEQAConfig(EQAConfig):
    """MP3D EQA specific configuration"""
    # MP3D specific question types
    supported_question_types: List[str] = field(default_factory=lambda: [
        "existence", "counting", "spatial", "color", "object_category", "room_type"
    ])

    # Answer types
    answer_types: List[str] = field(default_factory=lambda: [
        "yes/no", "number", "color", "object", "room", "other"
    ])

    # Navigation settings for EQA
    exploration_reward: float = 0.01
    look_around_bonus: float = 0.05


@dataclass
class EQASensorConfig:
    """EQA sensor configuration"""
    resolution: tuple = (224, 224)
    camera_height: float = 1.25
    max_question_length: int = 256
    vocab_size: int = 3000
    include_look_actions: bool = True


@dataclass
class EQAEvaluationConfig:
    """EQA evaluation configuration"""
    question_types: List[str] = field(default_factory=lambda: [
        "existence", "counting", "spatial", "color", "object_category"
    ])
    calculate_type_accuracy: bool = True
    calculate_mrr: bool = True
    save_predictions: bool = False
    output_file: Optional[str] = None