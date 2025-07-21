"""
Action space definitions for embodied agents
"""
import habitat_sim
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class ActionType(Enum):
    """Standard action types"""
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    STOP = "stop"
    ANSWER = "answer"
    TELEPORT = "teleport"


@dataclass
class ActionSpec:
    """Action specification"""
    name: str
    action_type: ActionType
    actuation_spec: Optional[habitat_sim.agent.ActuationSpec] = None
    parameters: Optional[Dict[str, Any]] = None


class BaseActionSpace(ABC):
    """Abstract base class for action spaces"""

    def __init__(self):
        self.actions: Dict[str, ActionSpec] = {}
        self._build_action_space()

    @abstractmethod
    def _build_action_space(self):
        """Build the action space"""
        pass

    def get_action_space(self) -> Dict[str, habitat_sim.agent.ActionSpec]:
        """Get habitat-sim compatible action space"""
        habitat_actions = {}
        for name, spec in self.actions.items():
            if spec.actuation_spec is not None:
                habitat_actions[name] = habitat_sim.agent.ActionSpec(
                    spec.action_type.value,
                    spec.actuation_spec
                )
        return habitat_actions

    def get_action_names(self) -> List[str]:
        """Get list of action names"""
        return list(self.actions.keys())

    def get_action_spec(self, action_name: str) -> Optional[ActionSpec]:
        """Get action specification by name"""
        return self.actions.get(action_name)


class DiscreteNavigationActionSpace(BaseActionSpace):
    """Standard discrete navigation action space"""

    def __init__(self,
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0,
                 tilt_angle: float = 30.0):
        self.forward_step = forward_step
        self.turn_angle = turn_angle
        self.tilt_angle = tilt_angle
        super().__init__()

    def _build_action_space(self):
        """Build discrete navigation actions"""
        self.actions = {
            ActionType.MOVE_FORWARD.value: ActionSpec(
                name="move_forward",
                action_type=ActionType.MOVE_FORWARD,
                actuation_spec=habitat_sim.agent.ActuationSpec(amount=self.forward_step)
            ),
            ActionType.TURN_LEFT.value: ActionSpec(
                name="turn_left",
                action_type=ActionType.TURN_LEFT,
                actuation_spec=habitat_sim.agent.ActuationSpec(amount=self.turn_angle)
            ),
            ActionType.TURN_RIGHT.value: ActionSpec(
                name="turn_right",
                action_type=ActionType.TURN_RIGHT,
                actuation_spec=habitat_sim.agent.ActuationSpec(amount=self.turn_angle)
            ),
            ActionType.STOP.value: ActionSpec(
                name="stop",
                action_type=ActionType.STOP,
                actuation_spec=habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }


class DiscreteNavigationWithLookActionSpace(DiscreteNavigationActionSpace):
    """Navigation action space with look up/down"""

    def _build_action_space(self):
        """Build navigation + look actions"""
        super()._build_action_space()

        # Add look actions
        self.actions.update({
            ActionType.LOOK_UP.value: ActionSpec(
                name="look_up",
                action_type=ActionType.LOOK_UP,
                actuation_spec=habitat_sim.agent.ActuationSpec(amount=self.tilt_angle)
            ),
            ActionType.LOOK_DOWN.value: ActionSpec(
                name="look_down",
                action_type=ActionType.LOOK_DOWN,
                actuation_spec=habitat_sim.agent.ActuationSpec(amount=self.tilt_angle)
            )
        })


class VLNActionSpace(DiscreteNavigationActionSpace):
    """Action space for Vision-Language Navigation"""

    def _build_action_space(self):
        """Build VLN-specific actions"""
        super()._build_action_space()

        # VLN typically uses the basic navigation actions
        # Stop action is particularly important for VLN
        pass


class EQAActionSpace(DiscreteNavigationWithLookActionSpace):
    """Action space for Embodied Question Answering"""

    def __init__(self,
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0,
                 tilt_angle: float = 30.0,
                 answer_vocab_size: int = 3000):
        self.answer_vocab_size = answer_vocab_size
        super().__init__(forward_step, turn_angle, tilt_angle)

    def _build_action_space(self):
        """Build EQA-specific actions"""
        super()._build_action_space()

        # Add answer action (discrete vocabulary)
        self.actions[ActionType.ANSWER.value] = ActionSpec(
            name="answer",
            action_type=ActionType.ANSWER,
            parameters={"vocab_size": self.answer_vocab_size}
        )


class ObjectNavActionSpace(DiscreteNavigationActionSpace):
    """Action space for Object Navigation"""

    def _build_action_space(self):
        """Build ObjectNav-specific actions"""
        super()._build_action_space()

        # ObjectNav uses standard navigation + stop
        # Stop action indicates object found
        pass


class ContinuousNavigationActionSpace(BaseActionSpace):
    """Continuous navigation action space"""

    def __init__(self,
                 max_forward: float = 0.5,
                 max_turn: float = 45.0):
        self.max_forward = max_forward
        self.max_turn = max_turn
        super().__init__()

    def _build_action_space(self):
        """Build continuous navigation actions"""
        # Continuous actions are handled differently
        # This would require custom action handling
        self.actions = {
            "continuous_nav": ActionSpec(
                name="continuous_nav",
                action_type=ActionType.MOVE_FORWARD,  # Base type
                parameters={
                    "max_forward": self.max_forward,
                    "max_turn": self.max_turn,
                    "action_dim": 2  # [forward_velocity, turn_velocity]
                }
            )
        }


class TeleportActionSpace(BaseActionSpace):
    """Action space with teleportation (for oracle navigation)"""

    def _build_action_space(self):
        """Build teleport actions"""
        self.actions = {
            ActionType.TELEPORT.value: ActionSpec(
                name="teleport",
                action_type=ActionType.TELEPORT,
                parameters={"requires_target_position": True}
            )
        }


class ActionSpaceFactory:
    """Factory for creating action spaces"""

    @staticmethod
    def create_action_space(action_space_type: str, **kwargs) -> BaseActionSpace:
        """Create action space by type"""
        action_space_type = action_space_type.lower()

        if action_space_type == "discrete_nav":
            return DiscreteNavigationActionSpace(**kwargs)
        elif action_space_type == "discrete_nav_look":
            return DiscreteNavigationWithLookActionSpace(**kwargs)
        elif action_space_type == "vln":
            return VLNActionSpace(**kwargs)
        elif action_space_type == "eqa":
            return EQAActionSpace(**kwargs)
        elif action_space_type == "objectnav":
            return ObjectNavActionSpace(**kwargs)
        elif action_space_type == "continuous_nav":
            return ContinuousNavigationActionSpace(**kwargs)
        elif action_space_type == "teleport":
            return TeleportActionSpace(**kwargs)
        else:
            raise ValueError(f"Unknown action space type: {action_space_type}")


# Utility functions for action handling
def action_to_index(action_name: str, action_space: BaseActionSpace) -> int:
    """Convert action name to index"""
    action_names = action_space.get_action_names()
    try:
        return action_names.index(action_name)
    except ValueError:
        raise ValueError(f"Action {action_name} not found in action space")


def index_to_action(index: int, action_space: BaseActionSpace) -> str:
    """Convert action index to name"""
    action_names = action_space.get_action_names()
    if 0 <= index < len(action_names):
        return action_names[index]
    else:
        raise ValueError(f"Action index {index} out of range")


# Pre-defined action spaces for common tasks
STANDARD_NAV_ACTIONS = DiscreteNavigationActionSpace()
VLN_ACTIONS = VLNActionSpace()
EQA_ACTIONS = EQAActionSpace()
OBJECTNAV_ACTIONS = ObjectNavActionSpace()