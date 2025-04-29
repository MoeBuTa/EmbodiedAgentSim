import argparse

import habitat_sim
import os

from tqdm import tqdm

import habitat
from easim.sim.simulator import set_sim, make_cfg
from easim.utils.config import hm3d_config
from easim.utils.constants import HM3D_SCENE_DIR

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser()
    return parser


def main():
    sim_settings = set_sim()
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])





