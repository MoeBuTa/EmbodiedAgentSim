import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1].resolve().parents[0]

HM3D_CONFIG_PATH = os.path.join(PROJECT_DIR, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
MP3D_CONFIG_PATH = os.path.join(PROJECT_DIR, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml")
R2R_CONFIG_PATH = os.path.join(PROJECT_DIR, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/vln_r2r.yaml")

DATA_PATH = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_PATH, "output")
os.mkdir(OUTPUT_DIR) if not os.path.exists(OUTPUT_DIR) else None




HM3D_SCENE_DIR = os.path.join(PROJECT_DIR, "data", "scene_datasets", "hm3d")
TEST_SCENE = os.path.join(HM3D_SCENE_DIR, "minival", "00800-TEEsavR23oF", "TEEsavR23oF.basis.glb")
TEST_HM3D_TASK = os.path.join(PROJECT_DIR, "data", "datasets", "objectnav_hm3d_v2", "val_mini", "val_mini.json.gz")

