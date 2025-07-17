from typing import Optional
from easim.utils.constants import (
    HM3D_CONFIG_PATH, MP3D_CONFIG_PATH, R2R_CONFIG_PATH,
    DATA_PATH, HM3D_SCENE_DIR, MP3D_SCENE_DIR
)


def get_basic_habitat_config():
    """Get a basic habitat config that works without external files"""
    try:
        import habitat
        from habitat.config.read_write import read_write
        from habitat.config.default_structured_configs import (
            CollisionsMeasurementConfig,
            FogOfWarConfig,
            TopDownMapMeasurementConfig,
        )

        # Try to create a minimal working config
        config = habitat.get_config()

        with read_write(config):
            # Basic dataset settings that work with local data
            config.habitat.dataset.scenes_dir = str(DATA_PATH / "scene_datasets")

            # Task measurements
            config.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=512,
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

            # Sensor settings
            if hasattr(config.habitat.simulator.agents.main_agent.sim_sensors, 'depth_sensor'):
                config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
                config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False

            # Task settings
            if hasattr(config.habitat.task.measurements, 'success'):
                config.habitat.task.measurements.success.success_distance = 0.25

        return config

    except ImportError:
        print("Warning: habitat module not available")
        return None
    except Exception as e:
        print(f"Warning: Could not create habitat config: {e}")
        return None


def hm3d_config(stage: str = 'val', episodes: int = 200):
    """Get HM3D config if available"""
    if HM3D_CONFIG_PATH is None or not HM3D_CONFIG_PATH.exists():
        print("Warning: HM3D config not found, using basic config")
        return get_basic_habitat_config()

    try:
        import habitat
        from habitat.config.read_write import read_write

        habitat_config = habitat.get_config(str(HM3D_CONFIG_PATH))

        with read_write(habitat_config):
            habitat_config.habitat.dataset.split = stage
            habitat_config.habitat.dataset.scenes_dir = str(HM3D_SCENE_DIR)
            habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

            # Use local scene dataset if available
            if (HM3D_SCENE_DIR / "hm3d_annotated_basis.scene_dataset_config.json").exists():
                habitat_config.habitat.simulator.scene_dataset = str(
                    HM3D_SCENE_DIR / "hm3d_annotated_basis.scene_dataset_config.json"
                )

        return habitat_config

    except Exception as e:
        print(f"Warning: Could not load HM3D config: {e}")
        return get_basic_habitat_config()


def mp3d_config(stage: str = 'val', episodes: int = 200):
    """Get MP3D config if available"""
    if MP3D_CONFIG_PATH is None or not MP3D_CONFIG_PATH.exists():
        print("Warning: MP3D config not found, using basic config")
        return get_basic_habitat_config()

    try:
        import habitat
        from habitat.config.read_write import read_write

        habitat_config = habitat.get_config(str(MP3D_CONFIG_PATH))

        with read_write(habitat_config):
            habitat_config.habitat.dataset.split = stage
            habitat_config.habitat.dataset.scenes_dir = str(MP3D_SCENE_DIR.parent)
            habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

            # Use local scene dataset if available
            if (MP3D_SCENE_DIR / "mp3d.scene_dataset_config.json").exists():
                habitat_config.habitat.simulator.scene_dataset = str(
                    MP3D_SCENE_DIR / "mp3d.scene_dataset_config.json"
                )

        return habitat_config

    except Exception as e:
        print(f"Warning: Could not load MP3D config: {e}")
        return get_basic_habitat_config()


def r2r_config(stage: str = 'val_seen', episodes: int = 200):
    """Get R2R config if available"""
    if R2R_CONFIG_PATH is None or not R2R_CONFIG_PATH.exists():
        print("Warning: R2R config not found, using basic config")
        return get_basic_habitat_config()

    try:
        import habitat
        from habitat.config.read_write import read_write

        habitat_config = habitat.get_config(str(R2R_CONFIG_PATH))

        with read_write(habitat_config):
            habitat_config.habitat.dataset.split = stage
            habitat_config.habitat.dataset.scenes_dir = str(MP3D_SCENE_DIR.parent)
            habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

            # Use local scene dataset
            if (MP3D_SCENE_DIR / "mp3d.scene_dataset_config.json").exists():
                habitat_config.habitat.simulator.scene_dataset = str(
                    MP3D_SCENE_DIR / "mp3d.scene_dataset_config.json"
                )

        return habitat_config

    except Exception as e:
        print(f"Warning: Could not load R2R config: {e}")
        return get_basic_habitat_config()