"""
Constants and paths for EmbodiedAgentSim
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_PATH / "output"

# Create basic directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SCENE DATASETS
# ============================================================================

# Scene dataset directories
SCENE_DATASETS_DIR = DATA_PATH / "scene_datasets"
HM3D_SCENE_DIR = SCENE_DATASETS_DIR / "hm3d"
MP3D_SCENE_DIR = SCENE_DATASETS_DIR / "mp3d"
MP3D_EXAMPLE_DIR = SCENE_DATASETS_DIR / "mp3d_example"

# Test scenes
TEST_SCENE_MP3D = MP3D_EXAMPLE_DIR / "17DRP5sb8fy" / "17DRP5sb8fy.glb"
TEST_SCENE_HM3D = HM3D_SCENE_DIR / "minival" / "00800-TEEsavR23oF" / "TEEsavR23oF.basis.glb"

# Scene dataset configs
MP3D_SCENE_DATASET = MP3D_EXAMPLE_DIR / "mp3d.scene_dataset_config.json"
HM3D_SCENE_DATASET = HM3D_SCENE_DIR / "hm3d_annotated_basis.scene_dataset_config.json"


# ============================================================================
# HABITAT CONFIG PATHS
# ============================================================================

HABITAT_LAB_DIR = PROJECT_DIR / "habitat-lab"
if HABITAT_LAB_DIR.exists():
    CONFIG_BASE = HABITAT_LAB_DIR / "habitat-lab" / "habitat" / "config" / "benchmark" / "nav"
    HM3D_CONFIG_PATH = CONFIG_BASE / "objectnav" / "objectnav_hm3d.yaml"
    MP3D_CONFIG_PATH = CONFIG_BASE / "objectnav" / "objectnav_mp3d.yaml"
    R2R_CONFIG_PATH = CONFIG_BASE / "vln_r2r.yaml"
    EQA_CONFIG_PATH = HABITAT_LAB_DIR / "habitat-lab" / "habitat" / "config" / "benchmark" / "eqa" / "mp3d_eqa.yaml"
else:
    HM3D_CONFIG_PATH = None
    MP3D_CONFIG_PATH = None
    R2R_CONFIG_PATH = None
    EQA_CONFIG_PATH = None


# ============================================================================
# TASK PARAMETERS
# ============================================================================

# Episode limits
MAX_EPISODE_STEPS = {
    "pointnav": 500,
    "objectnav": 500,
    "vln": 1000,
    "eqa": 200,
}

# Success thresholds
SUCCESS_DISTANCE = {
    "pointnav": 0.36,
    "objectnav": 1.0,
    "vln": 3.0,
    "eqa": 0.5,
}


# ============================================================================
# SIMULATOR SETTINGS
# ============================================================================

# Sensor settings
DEFAULT_SENSOR_RESOLUTION = (256, 256)
DEFAULT_HIGH_RES = (512, 512)
DEFAULT_VLN_RES = (224, 224)
DEFAULT_SENSOR_HEIGHT = 1.25

# Movement settings
DEFAULT_FORWARD_STEP = 0.25
DEFAULT_TURN_ANGLE = 30.0
DEFAULT_VLN_TURN_ANGLE = 15.0
DEFAULT_TILT_ANGLE = 30.0

# GPU settings
DEFAULT_GPU_DEVICE = 0
ENABLE_PHYSICS_DEFAULT = False


# ============================================================================
# ACTIONS
# ============================================================================

# Standard actions
ACTION_MOVE_FORWARD = "move_forward"
ACTION_TURN_LEFT = "turn_left"
ACTION_TURN_RIGHT = "turn_right"
ACTION_LOOK_UP = "look_up"
ACTION_LOOK_DOWN = "look_down"
ACTION_STOP = "stop"
ACTION_ANSWER = "answer"

# Action sets
BASIC_NAV_ACTIONS = [ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT]
VLN_ACTIONS = BASIC_NAV_ACTIONS + [ACTION_STOP]
EQA_ACTIONS = BASIC_NAV_ACTIONS + [ACTION_LOOK_UP, ACTION_LOOK_DOWN, ACTION_ANSWER]


# ============================================================================
# SENSORS
# ============================================================================

# Sensor names
SENSOR_RGB = "rgb"
SENSOR_COLOR = "color_sensor"
SENSOR_DEPTH = "depth"
SENSOR_SEMANTIC = "semantic"
SENSOR_GPS = "gps"
SENSOR_COMPASS = "compass"
SENSOR_POINTGOAL = "pointgoal_with_gps_compass"
SENSOR_OBJECTGOAL = "objectgoal"
SENSOR_INSTRUCTION = "instruction"
SENSOR_QUESTION = "question"


# ============================================================================
# OBJECT CATEGORIES
# ============================================================================

HM3D_OBJECTNAV_CATEGORIES = [
    "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
    "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
    "tv_monitor", "shower", "bathtub", "counter", "fireplace",
    "gym_equipment", "seating", "clothes"
]

MP3D_OBJECTNAV_CATEGORIES = [
    "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
    "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
    "tv_monitor", "shower", "bathtub"
]


# ============================================================================
# DATASET METADATA
# ============================================================================

# Dataset splits
R2R_SPLITS = ["train", "val_seen", "val_unseen", "test"]
OBJECTNAV_SPLITS = ["train", "val", "test"]
EQA_SPLITS = ["train", "val", "test"]

# EQA metadata
EQA_ANSWER_TYPES = ["yes/no", "number", "color", "object", "room", "other"]
EQA_QUESTION_TYPES = [
    "existence", "counting", "spatial", "color", "object_category",
    "room_type", "relative_position", "size", "material"
]


# ============================================================================
# VIDEO SETTINGS
# ============================================================================

DEFAULT_FPS = 30
DEFAULT_VIDEO_RESOLUTION = (640, 480)
DEFAULT_VIDEO_CODEC = 'mp4v'


# ============================================================================
# SUPPORTED TYPES
# ============================================================================

SUPPORTED_SCENE_DATASETS = ["MP3D", "HM3D"]
SUPPORTED_TASK_DATASETS = ["R2R", "ObjectNav", "EQA"]
SUPPORTED_TASKS = ["pointnav", "objectnav", "vln", "r2r", "eqa"]


# ============================================================================
# METRICS
# ============================================================================

NAVIGATION_METRICS = ["success_rate", "spl", "path_length", "distance_to_goal", "steps"]
VLN_METRICS = ["success_rate", "spl", "navigation_error", "path_length"]
EQA_METRICS = ["accuracy", "mean_reciprocal_rank"]



# ============================================================================
# DEFAULT CONFIGS
# ============================================================================

DEFAULT_TASK_CONFIGS = {
    "pointnav": {
        "max_episode_steps": 500,
        "success_distance": 0.36,
        "action_space_type": "discrete_nav",
        "sensor_suite_type": "pointnav"
    },
    "objectnav": {
        "max_episode_steps": 500,
        "success_distance": 1.0,
        "action_space_type": "objectnav",
        "sensor_suite_type": "objectnav"
    },
    "vln": {
        "max_episode_steps": 1000,
        "success_distance": 3.0,
        "action_space_type": "vln",
        "sensor_suite_type": "vln"
    },
    "eqa": {
        "max_episode_steps": 200,
        "success_distance": 0.5,
        "action_space_type": "eqa",
        "sensor_suite_type": "eqa"
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_scene_path(dataset_type: str, scene_name: str = None) -> Path:
    """Get scene path for dataset type"""
    dataset_type = dataset_type.upper()

    if dataset_type == "MP3D":
        if scene_name:
            candidates = [
                MP3D_SCENE_DIR / scene_name / f"{scene_name}.glb",
                MP3D_EXAMPLE_DIR / scene_name / f"{scene_name}.glb",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return TEST_SCENE_MP3D

    elif dataset_type == "HM3D":
        if scene_name:
            candidates = [
                HM3D_SCENE_DIR / "val" / scene_name / f"{scene_name}.basis.glb",
                HM3D_SCENE_DIR / "train" / scene_name / f"{scene_name}.basis.glb",
                HM3D_SCENE_DIR / "minival" / scene_name / f"{scene_name}.basis.glb",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return TEST_SCENE_HM3D

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_config_path(dataset_type: str) -> Path:
    """Get dataset config path"""
    dataset_type = dataset_type.upper()

    if dataset_type == "MP3D":
        if MP3D_SCENE_DATASET.exists():
            return MP3D_SCENE_DATASET
    elif dataset_type == "HM3D":
        if HM3D_SCENE_DATASET.exists():
            return HM3D_SCENE_DATASET

    return Path("")


def get_available_scenes(dataset_type: str) -> List[str]:
    """Get list of available scenes"""
    dataset_type = dataset_type.upper()
    scenes = []

    if dataset_type == "MP3D":
        for scene_dir in [MP3D_SCENE_DIR, MP3D_EXAMPLE_DIR]:
            if scene_dir.exists():
                for item in scene_dir.iterdir():
                    if item.is_dir():
                        scene_file = item / f"{item.name}.glb"
                        if scene_file.exists():
                            scenes.append(item.name)

    elif dataset_type == "HM3D":
        if HM3D_SCENE_DIR.exists():
            for split_dir in ["train", "val", "minival"]:
                split_path = HM3D_SCENE_DIR / split_dir
                if split_path.exists():
                    for item in split_path.iterdir():
                        if item.is_dir():
                            scene_file = item / f"{item.name}.basis.glb"
                            if scene_file.exists():
                                scenes.append(item.name)

    return sorted(list(set(scenes)))


def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        DATA_PATH, OUTPUT_DIR, SCENE_DATASETS_DIR, TASK_DATASETS_DIR,
        VLN_DATASETS_DIR, OBJECTNAV_DATASETS_DIR, EQA_DATASETS_DIR,
        R2R_DATASET_DIR, HM3D_OBJECTNAV_DIR, MP3D_OBJECTNAV_DIR, MP3D_EQA_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories
ensure_directories()