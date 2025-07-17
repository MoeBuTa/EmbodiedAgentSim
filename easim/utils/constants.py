import os
from pathlib import Path

# Project structure
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_PATH / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scene datasets
HM3D_SCENE_DIR = DATA_PATH / "scene_datasets" / "hm3d"
MP3D_SCENE_DIR = DATA_PATH / "scene_datasets" / "mp3d_example"

# Test scenes - use the paths that actually exist in your setup
TEST_SCENE_MP3D = MP3D_SCENE_DIR / "17DRP5sb8fy" / "17DRP5sb8fy.glb"
TEST_SCENE_HM3D = HM3D_SCENE_DIR / "minival" / "00800-TEEsavR23oF" / "TEEsavR23oF.basis.glb"

# Scene dataset configs - use relative paths that work with your setup
MP3D_SCENE_DATASET = MP3D_SCENE_DIR / "mp3d.scene_dataset_config.json"
HM3D_SCENE_DATASET = HM3D_SCENE_DIR / "hm3d_annotated_basis.scene_dataset_config.json"

# Habitat config paths (if they exist in your setup)
HABITAT_LAB_DIR = PROJECT_DIR / "habitat-lab"
if HABITAT_LAB_DIR.exists():
    CONFIG_BASE = HABITAT_LAB_DIR / "habitat-lab" / "habitat" / "config" / "benchmark" / "nav"
    HM3D_CONFIG_PATH = CONFIG_BASE / "objectnav" / "objectnav_hm3d.yaml"
    MP3D_CONFIG_PATH = CONFIG_BASE / "objectnav" / "objectnav_mp3d.yaml"
    R2R_CONFIG_PATH = CONFIG_BASE / "vln_r2r.yaml"
else:
    # Fallback - will be handled gracefully in config.py
    HM3D_CONFIG_PATH = None
    MP3D_CONFIG_PATH = None
    R2R_CONFIG_PATH = None

# Simulator defaults
DEFAULT_SENSOR_RESOLUTION = (256, 256)
DEFAULT_SENSOR_HEIGHT = 1.5
DEFAULT_FORWARD_STEP = 0.25
DEFAULT_TURN_ANGLE = 30.0

# Action names
ACTION_MOVE_FORWARD = "move_forward"
ACTION_TURN_LEFT = "turn_left"
ACTION_TURN_RIGHT = "turn_right"

# Sensor names
SENSOR_RGB = "color_sensor"
SENSOR_DEPTH = "depth_sensor"
SENSOR_SEMANTIC = "semantic_sensor"

# Video settings
DEFAULT_FPS = 30
DEFAULT_VIDEO_RESOLUTION = (640, 480)
DEFAULT_VIDEO_CODEC = 'mp4v'