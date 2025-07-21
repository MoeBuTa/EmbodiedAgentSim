"""
Object Navigation configuration dataclasses
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from easim.tasks.base.configs import NavigationConfig


@dataclass
class ObjectNavConfig(NavigationConfig):
    """ObjectNav specific configuration"""
    # Object categories and their mappings
    target_object_categories: List[str] = field(default_factory=lambda: [
        "chair", "table", "picture", "cabinet", "cushion", "sofa",
        "bed", "chest_of_drawers", "plant", "sink", "toilet",
        "stool", "towel", "tv_monitor", "shower", "bathtub",
        "counter", "fireplace", "gym_equipment", "seating", "clothes"
    ])
    object_category_mapping: Dict[str, int] = field(default_factory=dict)

    # Success criteria
    view_success: bool = True  # Success when object is in view vs. close to object
    view_success_distance: float = 1.0
    min_object_pixels: int = 50  # Minimum pixels for view success

    # Reward shaping
    view_reward_bonus: float = 1.0
    category_reward: float = 0.1  # Reward for seeing target category

    def __post_init__(self):
        super().__post_init__()

        if not self.object_category_mapping:
            self.object_category_mapping = {
                cat: i for i, cat in enumerate(self.target_object_categories)
            }

        # ObjectNav specific defaults
        self.action_space_type = "objectnav"
        self.sensor_suite_type = "objectnav"
        self.success_distance = 1.0


@dataclass
class HM3DObjectNavConfig(ObjectNavConfig):
    """HM3D ObjectNav specific configuration"""
    # HM3D specific object categories
    target_object_categories: List[str] = field(default_factory=lambda: [
        "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
        "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
        "tv_monitor", "shower", "bathtub", "counter", "fireplace",
        "gym_equipment", "seating", "clothes"
    ])

    # HM3D specific settings
    use_semantic_sensor: bool = True
    semantic_mapping_file: Optional[str] = None


@dataclass
class MP3DObjectNavConfig(ObjectNavConfig):
    """MP3D ObjectNav specific configuration"""
    # MP3D specific object categories (subset of HM3D)
    target_object_categories: List[str] = field(default_factory=lambda: [
        "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
        "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
        "tv_monitor", "shower", "bathtub"
    ])

    # MP3D specific settings
    use_gibson_semantics: bool = False