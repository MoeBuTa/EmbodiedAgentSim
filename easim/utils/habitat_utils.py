import pandas as pd
from datetime import datetime
from easim.utils.constants import (
    EVALUATION_DIR,
    HM3D_OBJECTNAV_CONFIG, MP3D_OBJECTNAV_CONFIG, MP3D_EQA_CONFIG, R2R_VLN_CONFIG,
    HM3D_SCENES_DIR, MP3D_SCENES_DIR,
    HM3D_OBJECTNAV_DATA, MP3D_OBJECTNAV_DATA, MP3D_EQA_DATA, R2R_VLN_DATA,
    HM3D_EQA_QUESTIONS, HM3D_EQA_SCENE_POSES, HM3D_EXPRESS_BENCH,
    HM3D_SCENE_DATASET, MP3D_SCENE_DATASET, SCENES_DIR
)

from typing import Dict, Optional, Any

import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)


def save_evaluation_results(task_name: str, metrics: Dict[str, float], num_episodes: int, agent_name: str = "unknown",
                            agent_model: str = "unknown") -> None:
    """
    Append evaluation results to a consolidated CSV file.
    
    :param task_name: Name of the task being evaluated
    :param metrics: Dictionary of evaluation metrics
    :param num_episodes: Number of episodes evaluated
    :param agent_name: Name/type of the agent being evaluated
    :param agent_model: Model used by the agent (e.g., gpt-4o-mini)
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "evaluation_results.csv"
    filepath = EVALUATION_DIR / filename

    # Prepare data for CSV
    data = {
        'task_name': [task_name],
        'agent_name': [agent_name],
        'agent_model': [agent_model],
        'num_episodes': [num_episodes],
        'timestamp': [timestamp]
    }

    # Add all metrics as columns
    for metric_name, metric_value in metrics.items():
        data[metric_name] = [metric_value]

    # Create DataFrame for new row
    new_row = pd.DataFrame(data)

    # Check if file exists and append or create
    if filepath.exists():
        # Read existing data to get all columns
        existing_df = pd.read_csv(filepath)
        # Combine with new row, filling missing columns with NaN
        combined_df = pd.concat([existing_df, new_row], ignore_index=True, sort=False)
        combined_df.to_csv(filepath, index=False)
    else:
        # Create new file
        new_row.to_csv(filepath, index=False)


def get_habitat_config(task_name: str, stage: str = 'val', episodes: int = 200):
    """
    Get habitat configuration based on task name using custom config functions.
    
    :param task_name: Name of the task (e.g., 'objectnav_hm3d', 'eqa_mp3d', etc.)
    :param stage: Dataset split for tasks that support it
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    # Map task names to their respective config functions
    config_mapping = {
        'objectnav_hm3d': lambda: objectnav_hm3d_config(stage, episodes),
        'objectnav_mp3d': lambda: objectnav_mp3d_config(stage, episodes),
        'eqa_mp3d': lambda: eqa_mp3d_config(stage, episodes),
        'eqa_hm3d': lambda: eqa_hm3d_config(episodes),
        'eqa_hm3d_express': lambda: eqa_hm3d_express_config(episodes),
        'vln_r2r': lambda: vln_r2r_config(stage, episodes),
    }

    return config_mapping[task_name]()


# ============================================================================
# CUSTOM HABITAT CONFIG FUNCTIONS
# ============================================================================

def objectnav_hm3d_config(stage: str = 'val', episodes: int = 200):
    """
    Create HM3D ObjectNav configuration with custom settings.
    
    :param stage: Dataset split ('train', 'val', 'minival')
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    habitat_config = habitat.get_config(HM3D_OBJECTNAV_CONFIG)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENES_DIR
        habitat_config.habitat.dataset.data_path = HM3D_OBJECTNAV_DATA.format(split=stage)
        habitat_config.habitat.simulator.scene_dataset = HM3D_SCENE_DATASET
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        habitat_config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25

    return habitat_config


def objectnav_mp3d_config(stage: str = 'val', episodes: int = 200):
    """
    Create MP3D ObjectNav configuration with custom settings.
    
    :param stage: Dataset split ('train', 'val', 'test')
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    habitat_config = habitat.get_config(MP3D_OBJECTNAV_CONFIG)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENES_DIR
        habitat_config.habitat.dataset.data_path = MP3D_OBJECTNAV_DATA.format(split=stage)
        habitat_config.habitat.simulator.scene_dataset = MP3D_SCENE_DATASET
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        habitat_config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25

    return habitat_config


def eqa_mp3d_config(stage: str = 'val', episodes: int = 200):
    """
    Create MP3D EQA configuration with custom settings.
    
    :param stage: Dataset split ('train', 'val', 'val_seen', 'val_unseen')
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    habitat_config = habitat.get_config(MP3D_EQA_CONFIG)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENES_DIR
        habitat_config.habitat.dataset.data_path = MP3D_EQA_DATA.format(split=stage)
        habitat_config.habitat.simulator.scene_dataset = MP3D_SCENE_DATASET
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        habitat_config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False

    return habitat_config


def eqa_hm3d_config(episodes: int = 200, **kwargs):
    """
    Create HM3D EQA configuration with custom settings.
    Note: HM3D EQA doesn't have train/val splits, uses direct file paths.
    
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    # Use MP3D EQA as base config and modify for HM3D
    habitat_config = habitat.get_config(MP3D_EQA_CONFIG)
    with read_write(habitat_config):
        # HM3D EQA specific configurations
        habitat_config.habitat.dataset.scenes_dir = SCENES_DIR
        habitat_config.habitat.simulator.scene_dataset = HM3D_SCENE_DATASET
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        # Set HM3D EQA specific file paths
        habitat_config.habitat.dataset.questions_path = HM3D_EQA_QUESTIONS
        habitat_config.habitat.dataset.scene_poses_path = HM3D_EQA_SCENE_POSES

        habitat_config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False

    return habitat_config


def eqa_hm3d_express_config(episodes: int = 200, **kwargs):
    """
    Create HM3D Express EQA configuration with custom settings.
    
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    habitat_config = eqa_hm3d_config(episodes)
    with read_write(habitat_config):
        # Set Express bench specific file path
        habitat_config.habitat.dataset.express_bench_path = HM3D_EXPRESS_BENCH

    return habitat_config


def vln_r2r_config(stage: str = 'val_seen', episodes: int = 200):
    """
    Create R2R VLN configuration with custom settings.
    
    :param stage: Dataset split ('train', 'val_seen', 'val_unseen')
    :param episodes: Number of episodes to sample
    :return: Configured habitat config object
    """
    habitat_config = habitat.get_config(R2R_VLN_CONFIG)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENES_DIR
        habitat_config.habitat.dataset.data_path = R2R_VLN_DATA.format(split=stage)
        habitat_config.habitat.simulator.scene_dataset = MP3D_SCENE_DATASET
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        habitat_config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25

    return habitat_config
