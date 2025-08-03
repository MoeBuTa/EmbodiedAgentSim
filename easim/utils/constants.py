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
# VIDEO SETTINGS
# ============================================================================

DEFAULT_FPS = 30
DEFAULT_VIDEO_RESOLUTION = (640, 480)
DEFAULT_VIDEO_CODEC = 'mp4v'
