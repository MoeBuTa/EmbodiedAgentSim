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
# SIMULATOR SETTINGS
# ============================================================================

# Sensor settings
DEFAULT_SENSOR_RESOLUTION = (256, 256)
DEFAULT_SENSOR_HEIGHT = 1.25

# Movement settings
DEFAULT_FORWARD_STEP = 0.25
DEFAULT_TURN_ANGLE = 30.0


# ============================================================================
# VIDEO SETTINGS
# ============================================================================

DEFAULT_FPS = 30
DEFAULT_VIDEO_RESOLUTION = (640, 480)
DEFAULT_VIDEO_CODEC = 'mp4v'







