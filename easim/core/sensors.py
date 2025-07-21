"""
Sensor configurations for embodied agents
"""
import habitat_sim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class SensorType(Enum):
    """Standard sensor types"""
    RGB = "rgb"
    DEPTH = "depth"
    SEMANTIC = "semantic"
    GPS = "gps"
    COMPASS = "compass"
    POINTGOAL = "pointgoal_with_gps_compass"
    OBJECTGOAL = "objectgoal"
    INSTRUCTION = "instruction"
    QUESTION = "question"


@dataclass
class SensorConfig:
    """Sensor configuration"""
    uuid: str
    sensor_type: SensorType
    resolution: Tuple[int, int] = (256, 256)
    position: List[float] = None
    orientation: List[float] = None
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.position is None:
            self.position = [0.0, 1.25, 0.0]  # Default camera height
        if self.orientation is None:
            self.orientation = [0.0, 0.0, 0.0]


class BaseSensorSuite(ABC):
    """Abstract base class for sensor suites"""

    def __init__(self):
        self.sensors: List[SensorConfig] = []
        self._build_sensor_suite()

    @abstractmethod
    def _build_sensor_suite(self):
        """Build the sensor suite"""
        pass

    def get_habitat_sensor_specs(self) -> List[habitat_sim.CameraSensorSpec]:
        """Get habitat-sim compatible sensor specifications"""
        specs = []

        for sensor in self.sensors:
            if sensor.sensor_type in [SensorType.RGB, SensorType.DEPTH, SensorType.SEMANTIC]:
                spec = self._create_camera_sensor_spec(sensor)
                specs.append(spec)

        return specs

    def _create_camera_sensor_spec(self, sensor_config: SensorConfig) -> habitat_sim.CameraSensorSpec:
        """Create camera sensor specification"""
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_config.uuid
        spec.resolution = [sensor_config.resolution[1], sensor_config.resolution[0]]  # [height, width]
        spec.position = sensor_config.position
        spec.orientation = sensor_config.orientation
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Set sensor type
        if sensor_config.sensor_type == SensorType.RGB:
            spec.sensor_type = habitat_sim.SensorType.COLOR
        elif sensor_config.sensor_type == SensorType.DEPTH:
            spec.sensor_type = habitat_sim.SensorType.DEPTH
        elif sensor_config.sensor_type == SensorType.SEMANTIC:
            spec.sensor_type = habitat_sim.SensorType.SEMANTIC

        return spec

    def get_sensor_uuids(self) -> List[str]:
        """Get list of sensor UUIDs"""
        return [sensor.uuid for sensor in self.sensors]

    def get_sensor_config(self, uuid: str) -> Optional[SensorConfig]:
        """Get sensor configuration by UUID"""
        for sensor in self.sensors:
            if sensor.uuid == uuid:
                return sensor
        return None


class BasicVisionSensorSuite(BaseSensorSuite):
    """Basic vision sensors (RGB + Depth)"""

    def __init__(self,
                 resolution: Tuple[int, int] = (256, 256),
                 camera_height: float = 1.25):
        self.resolution = resolution
        self.camera_height = camera_height
        super().__init__()

    def _build_sensor_suite(self):
        """Build basic vision sensors"""
        self.sensors = [
            SensorConfig(
                uuid="rgb",
                sensor_type=SensorType.RGB,
                resolution=self.resolution,
                position=[0.0, self.camera_height, 0.0]
            ),
            SensorConfig(
                uuid="depth",
                sensor_type=SensorType.DEPTH,
                resolution=self.resolution,
                position=[0.0, self.camera_height, 0.0]
            )
        ]


class FullVisionSensorSuite(BasicVisionSensorSuite):
    """Full vision sensors (RGB + Depth + Semantic)"""

    def _build_sensor_suite(self):
        """Build full vision sensors"""
        super()._build_sensor_suite()

        # Add semantic sensor
        self.sensors.append(
            SensorConfig(
                uuid="semantic",
                sensor_type=SensorType.SEMANTIC,
                resolution=self.resolution,
                position=[0.0, self.camera_height, 0.0]
            )
        )


class NavigationSensorSuite(FullVisionSensorSuite):
    """Navigation sensors with GPS and Compass"""

    def _build_sensor_suite(self):
        """Build navigation sensors"""
        super()._build_sensor_suite()

        # Add GPS and Compass sensors
        self.sensors.extend([
            SensorConfig(
                uuid="gps",
                sensor_type=SensorType.GPS,
                parameters={"dimensionality": 2}
            ),
            SensorConfig(
                uuid="compass",
                sensor_type=SensorType.COMPASS,
                parameters={"dimensionality": 1}
            )
        ])


class PointNavSensorSuite(NavigationSensorSuite):
    """Point Navigation sensor suite"""

    def _build_sensor_suite(self):
        """Build PointNav sensors"""
        super()._build_sensor_suite()

        # Add PointGoal sensor
        self.sensors.append(
            SensorConfig(
                uuid="pointgoal_with_gps_compass",
                sensor_type=SensorType.POINTGOAL,
                parameters={"goal_format": "POLAR", "dimensionality": 2}
            )
        )


class ObjectNavSensorSuite(NavigationSensorSuite):
    """Object Navigation sensor suite"""

    def _build_sensor_suite(self):
        """Build ObjectNav sensors"""
        super()._build_sensor_suite()

        # Add ObjectGoal sensor
        self.sensors.append(
            SensorConfig(
                uuid="objectgoal",
                sensor_type=SensorType.OBJECTGOAL,
                parameters={"goal_spec": "TASK_CATEGORY_ID"}
            )
        )


class VLNSensorSuite(FullVisionSensorSuite):
    """Vision-Language Navigation sensor suite"""

    def __init__(self,
                 resolution: Tuple[int, int] = (224, 224),
                 camera_height: float = 1.25,
                 max_instruction_length: int = 512):
        self.max_instruction_length = max_instruction_length
        super().__init__(resolution, camera_height)

    def _build_sensor_suite(self):
        """Build VLN sensors"""
        super()._build_sensor_suite()

        # Add instruction sensor
        self.sensors.append(
            SensorConfig(
                uuid="instruction",
                sensor_type=SensorType.INSTRUCTION,
                parameters={
                    "max_length": self.max_instruction_length,
                    "tokenizer": "clip"  # or "bert"
                }
            )
        )


class EQASensorSuite(FullVisionSensorSuite):
    """Embodied Question Answering sensor suite"""

    def __init__(self,
                 resolution: Tuple[int, int] = (224, 224),
                 camera_height: float = 1.25,
                 max_question_length: int = 256,
                 vocab_size: int = 3000):
        self.max_question_length = max_question_length
        self.vocab_size = vocab_size
        super().__init__(resolution, camera_height)

    def _build_sensor_suite(self):
        """Build EQA sensors"""
        super()._build_sensor_suite()

        # Add question sensor
        self.sensors.append(
            SensorConfig(
                uuid="question",
                sensor_type=SensorType.QUESTION,
                parameters={
                    "max_length": self.max_question_length,
                    "vocab_size": self.vocab_size
                }
            )
        )


class MultiViewSensorSuite(BaseSensorSuite):
    """Multi-view sensor suite with multiple cameras"""

    def __init__(self,
                 views: List[str] = ["front", "left", "right", "back"],
                 resolution: Tuple[int, int] = (256, 256),
                 camera_height: float = 1.25):
        self.views = views
        self.resolution = resolution
        self.camera_height = camera_height
        super().__init__()

    def _build_sensor_suite(self):
        """Build multi-view sensors"""
        # Orientations for different views (in degrees)
        orientations = {
            "front": [0.0, 0.0, 0.0],
            "left": [0.0, 90.0, 0.0],
            "right": [0.0, -90.0, 0.0],
            "back": [0.0, 180.0, 0.0],
            "up": [-90.0, 0.0, 0.0],
            "down": [90.0, 0.0, 0.0]
        }

        for view in self.views:
            if view in orientations:
                # Add RGB sensor for this view
                self.sensors.append(
                    SensorConfig(
                        uuid=f"rgb_{view}",
                        sensor_type=SensorType.RGB,
                        resolution=self.resolution,
                        position=[0.0, self.camera_height, 0.0],
                        orientation=orientations[view]
                    )
                )

                # Add depth sensor for this view
                self.sensors.append(
                    SensorConfig(
                        uuid=f"depth_{view}",
                        sensor_type=SensorType.DEPTH,
                        resolution=self.resolution,
                        position=[0.0, self.camera_height, 0.0],
                        orientation=orientations[view]
                    )
                )


class HighResSensorSuite(BasicVisionSensorSuite):
    """High resolution sensor suite"""

    def __init__(self,
                 resolution: Tuple[int, int] = (512, 512),
                 camera_height: float = 1.25):
        super().__init__(resolution, camera_height)


class SensorSuiteFactory:
    """Factory for creating sensor suites"""

    @staticmethod
    def create_sensor_suite(suite_type: str, **kwargs) -> BaseSensorSuite:
        """Create sensor suite by type"""
        suite_type = suite_type.lower()

        if suite_type == "basic_vision":
            return BasicVisionSensorSuite(**kwargs)
        elif suite_type == "full_vision":
            return FullVisionSensorSuite(**kwargs)
        elif suite_type == "navigation":
            return NavigationSensorSuite(**kwargs)
        elif suite_type == "pointnav":
            return PointNavSensorSuite(**kwargs)
        elif suite_type == "objectnav":
            return ObjectNavSensorSuite(**kwargs)
        elif suite_type == "vln":
            return VLNSensorSuite(**kwargs)
        elif suite_type == "eqa":
            return EQASensorSuite(**kwargs)
        elif suite_type == "multiview":
            return MultiViewSensorSuite(**kwargs)
        elif suite_type == "highres":
            return HighResSensorSuite(**kwargs)
        else:
            raise ValueError(f"Unknown sensor suite type: {suite_type}")


# Pre-defined sensor suites for common tasks
BASIC_VISION_SENSORS = BasicVisionSensorSuite()
FULL_VISION_SENSORS = FullVisionSensorSuite()
POINTNAV_SENSORS = PointNavSensorSuite()
OBJECTNAV_SENSORS = ObjectNavSensorSuite()
VLN_SENSORS = VLNSensorSuite()
EQA_SENSORS = EQASensorSuite()


def get_task_sensor_suite(task_type: str) -> BaseSensorSuite:
    """Get appropriate sensor suite for task type"""
    task_type = task_type.lower()

    if task_type in ["pointnav", "point_navigation"]:
        return POINTNAV_SENSORS
    elif task_type in ["objectnav", "object_navigation"]:
        return OBJECTNAV_SENSORS
    elif task_type in ["vln", "vision_language_navigation", "r2r"]:
        return VLN_SENSORS
    elif task_type in ["eqa", "embodied_qa", "embodied_question_answering"]:
        return EQA_SENSORS
    else:
        return FULL_VISION_SENSORS  # Default fallback