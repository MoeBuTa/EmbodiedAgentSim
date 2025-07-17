import os
import numpy as np
import habitat_sim
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from easim.utils.constants import (
    TEST_SCENE_MP3D, TEST_SCENE_HM3D,
    MP3D_SCENE_DATASET, HM3D_SCENE_DATASET,
    DEFAULT_SENSOR_RESOLUTION, DEFAULT_SENSOR_HEIGHT,
    DEFAULT_FORWARD_STEP, DEFAULT_TURN_ANGLE,
    ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
    SENSOR_RGB, SENSOR_DEPTH, SENSOR_SEMANTIC
)


@dataclass
class SimulatorConfig:
    """Configuration for simulator setup"""
    scene_path: str
    scene_dataset: str
    width: int = DEFAULT_SENSOR_RESOLUTION[0]
    height: int = DEFAULT_SENSOR_RESOLUTION[1]
    sensor_height: float = DEFAULT_SENSOR_HEIGHT
    enable_physics: bool = False
    enable_rgb: bool = True
    enable_depth: bool = True
    enable_semantic: bool = True
    forward_step_size: float = DEFAULT_FORWARD_STEP
    turn_angle: float = DEFAULT_TURN_ANGLE
    gpu_device_id: int = 0


class BaseSensorManager:
    """Manages sensor configurations for the simulator"""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self._sensor_specs = []

    def create_sensor_specs(self) -> List[habitat_sim.CameraSensorSpec]:
        """Create sensor specifications based on config"""
        self._sensor_specs = []

        if self.config.enable_rgb:
            self._add_rgb_sensor()
        if self.config.enable_depth:
            self._add_depth_sensor()
        if self.config.enable_semantic:
            self._add_semantic_sensor()

        return self._sensor_specs

    def _add_rgb_sensor(self):
        """Add RGB camera sensor"""
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = SENSOR_RGB
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.config.height, self.config.width]
        rgb_spec.position = [0.0, self.config.sensor_height, 0.0]
        rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sensor_specs.append(rgb_spec)

    def _add_depth_sensor(self):
        """Add depth camera sensor"""
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = SENSOR_DEPTH
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [self.config.height, self.config.width]
        depth_spec.position = [0.0, self.config.sensor_height, 0.0]
        depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sensor_specs.append(depth_spec)

    def _add_semantic_sensor(self):
        """Add semantic camera sensor"""
        semantic_spec = habitat_sim.CameraSensorSpec()
        semantic_spec.uuid = SENSOR_SEMANTIC
        semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_spec.resolution = [self.config.height, self.config.width]
        semantic_spec.position = [0.0, self.config.sensor_height, 0.0]
        semantic_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sensor_specs.append(semantic_spec)


class BaseSimulator(ABC):
    """Abstract base class for habitat simulators"""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.sensor_manager = BaseSensorManager(config)
        self.sim = None
        self.agent = None
        self._initialize_simulator()

    def _initialize_simulator(self):
        """Initialize the habitat simulator"""
        cfg = self._make_habitat_config()
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.initialize_agent(0)

    def _make_habitat_config(self) -> habitat_sim.Configuration:
        """Create habitat simulator configuration"""
        # Simulator configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = self.config.gpu_device_id
        sim_cfg.scene_id = self.config.scene_path
        sim_cfg.scene_dataset_config_file = self.config.scene_dataset
        sim_cfg.enable_physics = self.config.enable_physics

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = self.sensor_manager.create_sensor_specs()
        agent_cfg.action_space = self._create_action_space()

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _create_action_space(self) -> Dict[str, habitat_sim.agent.ActionSpec]:
        """Create action space for the agent"""
        return {
            ACTION_MOVE_FORWARD: habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=float(self.config.forward_step_size))
            ),
            ACTION_TURN_LEFT: habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=float(self.config.turn_angle))
            ),
            ACTION_TURN_RIGHT: habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=float(self.config.turn_angle))
            ),
        }

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get current sensor observations"""
        if self.sim is None:
            raise RuntimeError("Simulator not initialized")
        return self.sim.get_sensor_observations()

    def step(self, action: str) -> Dict[str, np.ndarray]:
        """Take an action and return observations"""
        if self.sim is None:
            raise RuntimeError("Simulator not initialized")
        return self.sim.step(action)

    def get_agent_state(self) -> habitat_sim.AgentState:
        """Get current agent state"""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        return self.agent.get_state()

    def set_agent_state(self, position: List[float], rotation: List[float]):
        """Set agent position and rotation"""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")

        state = habitat_sim.AgentState()
        state.position = np.array(position, dtype=np.float32)
        # Skip rotation setting for now to avoid quaternion issues
        # state.rotation = np.quaternion(1, 0, 0, 0)  # Identity quaternion
        self.agent.set_state(state)

    def reset(self):
        """Reset the simulator"""
        if self.sim is not None:
            self.sim.reset()

    def close(self):
        """Close the simulator"""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    @abstractmethod
    def get_scene_info(self) -> Dict[str, Any]:
        """Get scene-specific information"""
        pass


class MP3DSimulator(BaseSimulator):
    """Simulator for MP3D scenes"""

    def __init__(self, scene_name: Optional[str] = None):
        # Use provided scene or default MP3D scene
        if scene_name:
            scene_path = scene_name
        else:
            scene_path = str(TEST_SCENE_MP3D)

        # Use MP3D scene dataset config if it exists
        if MP3D_SCENE_DATASET.exists():
            scene_dataset = str(MP3D_SCENE_DATASET)
        else:
            # Fallback to scene path for basic setup
            scene_dataset = ""

        config = SimulatorConfig(
            scene_path=scene_path,
            scene_dataset=scene_dataset
        )
        super().__init__(config)

    def get_scene_info(self) -> Dict[str, Any]:
        """Get MP3D scene information"""
        return {
            "dataset": "MP3D",
            "scene_path": self.config.scene_path,
            "scene_dataset": self.config.scene_dataset,
            "scene_name": Path(self.config.scene_path).stem
        }


class HM3DSimulator(BaseSimulator):
    """Simulator for HM3D scenes"""

    def __init__(self, scene_name: Optional[str] = None):
        # Use provided scene or default HM3D scene
        if scene_name:
            scene_path = scene_name
        else:
            scene_path = str(TEST_SCENE_HM3D)

        # Use HM3D scene dataset config if it exists
        if HM3D_SCENE_DATASET.exists():
            scene_dataset = str(HM3D_SCENE_DATASET)
        else:
            # Fallback to scene path for basic setup
            scene_dataset = ""

        config = SimulatorConfig(
            scene_path=scene_path,
            scene_dataset=scene_dataset
        )
        super().__init__(config)

    def get_scene_info(self) -> Dict[str, Any]:
        """Get HM3D scene information"""
        return {
            "dataset": "HM3D",
            "scene_path": self.config.scene_path,
            "scene_dataset": self.config.scene_dataset,
            "scene_name": Path(self.config.scene_path).stem
        }


class SimulatorFactory:
    """Factory for creating different types of simulators"""

    @staticmethod
    def create_simulator(dataset: str, scene_name: Optional[str] = None) -> BaseSimulator:
        """Create simulator based on dataset type"""
        dataset = dataset.upper()

        if dataset == "MP3D":
            return MP3DSimulator(scene_name)
        elif dataset == "HM3D":
            return HM3DSimulator(scene_name)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Use 'MP3D' or 'HM3D'")

    @staticmethod
    def create_custom_simulator(config: SimulatorConfig) -> BaseSimulator:
        """Create simulator with custom configuration"""

        class CustomSimulator(BaseSimulator):
            def get_scene_info(self) -> Dict[str, Any]:
                return {
                    "dataset": "Custom",
                    "scene_path": self.config.scene_path,
                    "scene_dataset": self.config.scene_dataset
                }

        return CustomSimulator(config)