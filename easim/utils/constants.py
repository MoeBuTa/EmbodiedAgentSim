"""
Constants and paths for EmbodiedAgentSim
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from easim.agents.eqa import EQAAgent
from easim.agents.objectnav import ObjectNavAgent
from easim.agents.sample import SampleAgent

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_PATH / "output"

# Create basic directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = OUTPUT_DIR / "videos"
# Create video directory if it doesn't exist
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = OUTPUT_DIR / "images"
# Create image directory if it doesn't exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_DIR = OUTPUT_DIR / "evaluations"
# Create evaluation directory if it doesn't exist
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)



CONFIG_DIR = PROJECT_DIR / "easim/config"

# ============================================================================
# BENCHMARK SETTINGS
# ============================================================================

BENCHMARK_CONFIG = [
    # Navigation benchmarks
    "objectnav_hm3d",
    "objectnav_mp3d",
    "eqa_mp3d",
    "eqa_hm3d",
    "eqa_hm3d_express",
    "vln_r2r",
    "pointnav_base",
    "pointnav_franka",
    "pointnav_gibson",
    "pointnav_habitat_test",
    "pointnav_hm3d",
    "pointnav_mp3d",
    "imagenav_base",
    "imagenav_gibson",
    "imagenav_mp3d",
    "imagenav_test",
    "instance_imagenav_hm3d_v1",
    "instance_imagenav_hm3d_v2",
    
    # Rearrange benchmarks
    "rearrange_idle",
    "rearrange_idle_single_camera",
    "rearrange_interact",
    "rearrange_prepare_groceries",
    "rearrange_multi_task",
    "rearrange_easy",
    "rearrange_set_table",
    "rearrange_tidy_house",
    "pddl_geo_base_task",
    "play",
    "play_human",
    "play_spot",
    "play_stretch",
    "skill_close_cab",
    "skill_close_fridge",
    "skill_nav_to_obj",
    "skill_open_cab",
    "skill_open_fridge",
    "skill_pick",
    "skill_pick_spot",
    "skill_place",
    "skill_reach_state",
    "hab3_bench_humanoid_oracle",
    "hab3_bench_multi_agent",
    "hab3_bench_single_agent",
    "hab3_bench_spot_humanoid_oracle",
    "hab3_bench_spot_oracle",
    "hab3_bench_spot_spot_oracle",
    
    # Multi-agent benchmarks
    "multi_agent_hssd_spot_human",
    "multi_agent_hssd_spot_human_social_nav",
    "multi_agent_social_nav",
    "multi_agent_tidy_house",
]



# ============================================================================
# VIDEO SETTINGS
# ============================================================================

DEFAULT_FPS = 30
DEFAULT_VIDEO_RESOLUTION = (640, 480)
DEFAULT_VIDEO_CODEC = 'mp4v'



# ============================================================================
# HABITAT CONFIG PATHS
# ============================================================================

# Base habitat-lab config directory
HABITAT_CONFIG_DIR = PROJECT_DIR / "habitat-lab" / "habitat-lab" / "habitat" / "config" / "benchmark"

# Task-specific config paths
HM3D_OBJECTNAV_CONFIG = f"{HABITAT_CONFIG_DIR}/nav/objectnav/objectnav_hm3d.yaml"
MP3D_OBJECTNAV_CONFIG = f"{HABITAT_CONFIG_DIR}/nav/objectnav/objectnav_mp3d.yaml"
MP3D_EQA_CONFIG = f"{HABITAT_CONFIG_DIR}/nav/eqa_mp3d.yaml"
R2R_VLN_CONFIG = f"{HABITAT_CONFIG_DIR}/nav/vln_r2r.yaml"

# Dataset paths  
SCENES_DIR = f"{DATA_PATH}/scene_datasets"
HM3D_SCENES_DIR = f"{DATA_PATH}/scene_datasets/hm3d_v0.2"
MP3D_SCENES_DIR = f"{DATA_PATH}/scene_datasets/mp3d"

# Task data paths
HM3D_OBJECTNAV_DATA = f"{DATA_PATH}/datasets/objectnav/hm3d/v2/{{split}}/{{split}}.json.gz"
MP3D_OBJECTNAV_DATA = f"{DATA_PATH}/datasets/objectnav/mp3d/v1/{{split}}/{{split}}.json.gz"
MP3D_EQA_DATA = f"{DATA_PATH}/datasets/eqa/mp3d/v1/{{split}}/{{split}}.json.gz"
R2R_VLN_DATA = f"{DATA_PATH}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz"

# HM3D EQA specific files (no train/val split)
HM3D_EQA_QUESTIONS = f"{DATA_PATH}/datasets/eqa/hm3d/questions.csv"
HM3D_EQA_SCENE_POSES = f"{DATA_PATH}/datasets/eqa/hm3d/scene_init_poses.csv"

# Express bench specific files
HM3D_EXPRESS_BENCH = f"{DATA_PATH}/datasets/eqa/hm3d/express-bench.json"

# Scene dataset config paths
HM3D_SCENE_DATASET = f"{HM3D_SCENES_DIR}/hm3d_annotated_basis.scene_dataset_config.json"
MP3D_SCENE_DATASET = f"{MP3D_SCENES_DIR}/mp3d.scene_dataset_config.json"


# ============================================================================
# AGENT SETTINGS
# ============================================================================

AGENT_LIST = {
    "sample": {
        "type": SampleAgent,
        "description": "A sample agent that performs random actions in the environment."
    },
    "objectnav": {
        "type": ObjectNavAgent,
        "description": "An agent designed for object navigation tasks, using LLMs to generate navigation prompts."
    },
    "eqa": {
        "type": EQAAgent,
        "description": "An agent for embodied question answering tasks, leveraging LLMs for understanding and navigation."
    },
}