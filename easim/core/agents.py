"""
Agent configurations for embodied simulation
"""
import habitat_sim
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from easim.core.actions import BaseActionSpace, ActionSpaceFactory
from easim.core.sensors import BaseSensorSuite, SensorSuiteFactory


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_id: int = 0
    height: float = 1.5
    radius: float = 0.2
    mass: float = 32.0
    angular_acceleration: float = 4 * np.pi
    linear_friction: float = 0.5
    angular_friction: float = 1.0
    coefficient_of_restitution: float = 0.0
    sensor_height: float = 1.25
    action_space_type: str = "discrete_nav"
    sensor_suite_type: str = "full_vision"
    action_space_config: Optional[Dict[str, Any]] = None
    sensor_suite_config: Optional[Dict[str, Any]] = None


class BaseAgent:
    """Base agent class"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.action_space = self._create_action_space()
        self.sensor_suite = self._create_sensor_suite()

    def _create_action_space(self) -> BaseActionSpace:
        """Create action space"""
        config = self.config.action_space_config or {}
        return ActionSpaceFactory.create_action_space(
            self.config.action_space_type, **config
        )

    def _create_sensor_suite(self) -> BaseSensorSuite:
        """Create sensor suite"""
        config = self.config.sensor_suite_config or {}
        config.setdefault('camera_height', self.config.sensor_height)
        return SensorSuiteFactory.create_sensor_suite(
            self.config.sensor_suite_type, **config
        )

    def get_habitat_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """Get habitat-sim compatible agent configuration"""
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # Basic agent properties
        agent_cfg.height = self.config.height
        agent_cfg.radius = self.config.radius

        # Sensor and action configurations
        agent_cfg.sensor_specifications = self.sensor_suite.get_habitat_sensor_specs()
        agent_cfg.action_space = self.action_space.get_action_space()

        return agent_cfg

    def get_action_names(self) -> List[str]:
        """Get available action names"""
        return self.action_space.get_action_names()

    def get_sensor_uuids(self) -> List[str]:
        """Get sensor UUIDs"""
        return self.sensor_suite.get_sensor_uuids()


class NavigationAgent(BaseAgent):
    """Agent for navigation tasks"""

    def __init__(self,
                 action_space_type: str = "discrete_nav",
                 sensor_suite_type: str = "navigation",
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0,
                 **kwargs):
        config = AgentConfig(
            action_space_type=action_space_type,
            sensor_suite_type=sensor_suite_type,
            action_space_config={
                "forward_step": forward_step,
                "turn_angle": turn_angle
            },
            **kwargs
        )
        super().__init__(config)


class PointNavAgent(NavigationAgent):
    """Agent for Point Navigation"""

    def __init__(self, **kwargs):
        kwargs.setdefault('sensor_suite_type', 'pointnav')
        super().__init__(**kwargs)


class ObjectNavAgent(NavigationAgent):
    """Agent for Object Navigation"""

    def __init__(self, **kwargs):
        kwargs.setdefault('action_space_type', 'objectnav')
        kwargs.setdefault('sensor_suite_type', 'objectnav')
        super().__init__(**kwargs)


class VLNAgent(BaseAgent):
    """Agent for Vision-Language Navigation"""

    def __init__(self,
                 forward_step: float = 0.25,
                 turn_angle: float = 15.0,  # Smaller turns for VLN
                 resolution: tuple = (224, 224),
                 max_instruction_length: int = 512,
                 **kwargs):
        config = AgentConfig(
            action_space_type="vln",
            sensor_suite_type="vln",
            action_space_config={
                "forward_step": forward_step,
                "turn_angle": turn_angle
            },
            sensor_suite_config={
                "resolution": resolution,
                "max_instruction_length": max_instruction_length
            },
            **kwargs
        )
        super().__init__(config)


class EQAAgent(BaseAgent):
    """Agent for Embodied Question Answering"""

    def __init__(self,
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0,
                 tilt_angle: float = 30.0,
                 resolution: tuple = (224, 224),
                 answer_vocab_size: int = 3000,
                 **kwargs):
        config = AgentConfig(
            action_space_type="eqa",
            sensor_suite_type="eqa",
            action_space_config={
                "forward_step": forward_step,
                "turn_angle": turn_angle,
                "tilt_angle": tilt_angle,
                "answer_vocab_size": answer_vocab_size
            },
            sensor_suite_config={
                "resolution": resolution,
                "vocab_size": answer_vocab_size
            },
            **kwargs
        )
        super().__init__(config)


class HighResAgent(BaseAgent):
    """Agent with high resolution sensors"""

    def __init__(self,
                 resolution: tuple = (512, 512),
                 **kwargs):
        config = AgentConfig(
            sensor_suite_type="highres",
            sensor_suite_config={"resolution": resolution},
            **kwargs
        )
        super().__init__(config)


class MultiViewAgent(BaseAgent):
    """Agent with multiple camera views"""

    def __init__(self,
                 views: List[str] = ["front", "left", "right"],
                 resolution: tuple = (256, 256),
                 **kwargs):
        config = AgentConfig(
            sensor_suite_type="multiview",
            sensor_suite_config={
                "views": views,
                "resolution": resolution
            },
            **kwargs
        )
        super().__init__(config)


class AgentFactory:
    """Factory for creating agents"""

    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> BaseAgent:
        """Create agent by type"""
        agent_type = agent_type.lower()

        if agent_type == "base":
            return BaseAgent(AgentConfig(**kwargs))
        elif agent_type == "navigation":
            return NavigationAgent(**kwargs)
        elif agent_type == "pointnav":
            return PointNavAgent(**kwargs)
        elif agent_type == "objectnav":
            return ObjectNavAgent(**kwargs)
        elif agent_type == "vln":
            return VLNAgent(**kwargs)
        elif agent_type == "eqa":
            return EQAAgent(**kwargs)
        elif agent_type == "highres":
            return HighResAgent(**kwargs)
        elif agent_type == "multiview":
            return MultiViewAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


# Pre-defined agents for common tasks
def get_task_agent(task_type: str, **kwargs) -> BaseAgent:
    """Get appropriate agent for task type"""
    task_type = task_type.lower()

    if task_type in ["pointnav", "point_navigation"]:
        return PointNavAgent(**kwargs)
    elif task_type in ["objectnav", "object_navigation"]:
        return ObjectNavAgent(**kwargs)
    elif task_type in ["vln", "vision_language_navigation", "r2r"]:
        return VLNAgent(**kwargs)
    elif task_type in ["eqa", "embodied_qa", "embodied_question_answering"]:
        return EQAAgent(**kwargs)
    else:
        return NavigationAgent(**kwargs)  # Default fallback


# Predefined common agent configurations
STANDARD_NAV_AGENT = NavigationAgent()
POINTNAV_AGENT = PointNavAgent()
OBJECTNAV_AGENT = ObjectNavAgent()
VLN_AGENT = VLNAgent()
EQA_AGENT = EQAAgent()