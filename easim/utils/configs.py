from easim.utils.constants import (
    SCENES_DIR, HABITAT_TASK_CONFIGS
)



import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)

def habitat_lab_config(task_name: str, stage: str = 'val', episodes: int = 200, **kwargs):
    """
    Unified habitat lab configuration function that supports all tasks.
    
    :param task_name: Name of the task ('objectnav_hm3d', 'objectnav_mp3d', 'eqa_mp3d', 'vln_r2r')
    :param stage: Dataset split (varies by task)
    :param episodes: Number of episodes to sample
    :param kwargs: Additional task-specific parameters
    :return: Configured habitat config object
    """
    if task_name not in HABITAT_TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {list(HABITAT_TASK_CONFIGS.keys())}")
    
    config_info = HABITAT_TASK_CONFIGS[task_name]
    habitat_config = habitat.get_config(config_info['config_file'])
    
    with read_write(habitat_config):
        # Common configuration for all tasks
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENES_DIR
        habitat_config.habitat.dataset.data_path = config_info['data_path'].format(split=stage)
        habitat_config.habitat.simulator.scene_dataset = config_info['scene_dataset']
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        
        # Common measurements for all tasks
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
        
        # Common sensor configurations
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        
        # Success distance for navigation tasks
        if config_info['has_success_distance']:
            habitat_config.habitat.task.measurements.success.success_distance = 0.25
    
    return habitat_config

