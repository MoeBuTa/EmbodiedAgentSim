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

VIDEO_DIR = OUTPUT_DIR / "videos"
# Create video directory if it doesn't exist
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = OUTPUT_DIR / "images"
# Create image directory if it doesn't exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# BENCHMARK SETTINGS
# ============================================================================

BENCHMARK_CONFIG = {
    # Navigation benchmarks
    "objectnav_hm3d": "benchmark/nav/objectnav/objectnav_hm3d.yaml",
    "objectnav_hm3d_with_semantic": "benchmark/nav/objectnav/objectnav_hm3d_with_semantic.yaml",
    "objectnav_hssd_hab": "benchmark/nav/objectnav/objectnav_hssd-hab.yaml",
    "objectnav_hssd_hab_with_semantic": "benchmark/nav/objectnav/objectnav_hssd-hab_with_semantic.yaml",
    "objectnav_mp3d": "benchmark/nav/objectnav/objectnav_mp3d.yaml",
    "objectnav_mp3d_with_semantic": "benchmark/nav/objectnav/objectnav_mp3d_with_semantic.yaml",
    "objectnav_procthor_hab": "benchmark/nav/objectnav/objectnav_procthor-hab.yaml",
    "objectnav_procthor_hab_with_semantic": "benchmark/nav/objectnav/objectnav_procthor-hab_with_semantic.yaml",
    
    "pointnav_base": "benchmark/nav/pointnav/pointnav_base.yaml",
    "pointnav_franka": "benchmark/nav/pointnav/pointnav_franka.yaml",
    "pointnav_gibson": "benchmark/nav/pointnav/pointnav_gibson.yaml",
    "pointnav_habitat_test": "benchmark/nav/pointnav/pointnav_habitat_test.yaml",
    "pointnav_hm3d": "benchmark/nav/pointnav/pointnav_hm3d.yaml",
    "pointnav_mp3d": "benchmark/nav/pointnav/pointnav_mp3d.yaml",
    
    "imagenav_base": "benchmark/nav/imagenav/imagenav_base.yaml",
    "imagenav_gibson": "benchmark/nav/imagenav/imagenav_gibson.yaml",
    "imagenav_mp3d": "benchmark/nav/imagenav/imagenav_mp3d.yaml",
    "imagenav_test": "benchmark/nav/imagenav/imagenav_test.yaml",
    
    "instance_imagenav_hm3d_v1": "benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v1.yaml",
    "instance_imagenav_hm3d_v2": "benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml",
    
    "eqa_mp3d": "benchmark/nav/eqa_mp3d.yaml",
    "eqa_rgbonly_mp3d": "benchmark/nav/eqa_rgbonly_mp3d.yaml",
    "vln_r2r": "benchmark/nav/vln_r2r.yaml",
    
    # Rearrange benchmarks
    "rearrange_idle": "benchmark/rearrange/demo/idle.yaml",
    "rearrange_idle_single_camera": "benchmark/rearrange/demo/idle_single_camera.yaml",
    "rearrange_interact": "benchmark/rearrange/demo/interact.yaml",
    
    "rearrange_prepare_groceries": "benchmark/rearrange/multi_task/prepare_groceries.yaml",
    "rearrange_multi_task": "benchmark/rearrange/multi_task/rearrange.yaml",
    "rearrange_easy": "benchmark/rearrange/multi_task/rearrange_easy.yaml",
    "rearrange_set_table": "benchmark/rearrange/multi_task/set_table.yaml",
    "rearrange_tidy_house": "benchmark/rearrange/multi_task/tidy_house.yaml",
    "pddl_geo_base_task": "benchmark/rearrange/multi_task/pddl_geo_base_task.yaml",
    
    "play": "benchmark/rearrange/play/play.yaml",
    "play_human": "benchmark/rearrange/play/play_human.yaml",
    "play_spot": "benchmark/rearrange/play/play_spot.yaml",
    "play_stretch": "benchmark/rearrange/play/play_stretch.yaml",
    
    "skill_close_cab": "benchmark/rearrange/skills/close_cab.yaml",
    "skill_close_fridge": "benchmark/rearrange/skills/close_fridge.yaml",
    "skill_nav_to_obj": "benchmark/rearrange/skills/nav_to_obj.yaml",
    "skill_open_cab": "benchmark/rearrange/skills/open_cab.yaml",
    "skill_open_fridge": "benchmark/rearrange/skills/open_fridge.yaml",
    "skill_pick": "benchmark/rearrange/skills/pick.yaml",
    "skill_pick_spot": "benchmark/rearrange/skills/pick_spot.yaml",
    "skill_place": "benchmark/rearrange/skills/place.yaml",
    "skill_reach_state": "benchmark/rearrange/skills/reach_state.yaml",
    
    "hab3_bench_humanoid_oracle": "benchmark/rearrange/hab3_bench/humanoid_oracle.yaml",
    "hab3_bench_multi_agent": "benchmark/rearrange/hab3_bench/multi_agent_bench.yaml",
    "hab3_bench_single_agent": "benchmark/rearrange/hab3_bench/single_agent_bench.yaml",
    "hab3_bench_spot_humanoid_oracle": "benchmark/rearrange/hab3_bench/spot_humanoid_oracle.yaml",
    "hab3_bench_spot_oracle": "benchmark/rearrange/hab3_bench/spot_oracle.yaml",
    "hab3_bench_spot_spot_oracle": "benchmark/rearrange/hab3_bench/spot_spot_oracle.yaml",
    
    # Multi-agent benchmarks
    "multi_agent_hssd_spot_human": "benchmark/multi_agent/hssd_spot_human.yaml",
    "multi_agent_hssd_spot_human_social_nav": "benchmark/multi_agent/hssd_spot_human_social_nav.yaml",
    "multi_agent_social_nav": "benchmark/multi_agent/pddl/multi_agent_social_nav.yaml",
    "multi_agent_tidy_house": "benchmark/multi_agent/pddl/multi_agent_tidy_house.yaml",
}



# ============================================================================
# VIDEO SETTINGS
# ============================================================================

DEFAULT_FPS = 30
DEFAULT_VIDEO_RESOLUTION = (640, 480)
DEFAULT_VIDEO_CODEC = 'mp4v'



# ============================================================================
# AGENT SETTINGS
# ============================================================================

AGENT_LIST = {
    "sample": {
        "type": "SampleAgent",
        "description": "A sample agent that performs random actions in the environment."
    },
    "objectnav": {
        "type": "ObjectNavAgent",
        "description": "An agent designed for object navigation tasks, using LLMs to generate navigation prompts."
    },
    "eqa": {
        "type": "EQAAgent",
        "description": "An agent for embodied question answering tasks, leveraging LLMs for understanding and navigation."
    },
}