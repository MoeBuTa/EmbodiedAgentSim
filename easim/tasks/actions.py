"""Custom action classes for extending habitat-lab functionality"""
from typing import Any, Dict
from abc import ABC, abstractmethod

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.spaces import EmptySpace


class BaseCustomAction(SimulatorTaskAction):
    """
    Base class for custom actions that extend habitat-lab's SimulatorTaskAction
    
    Use this when you need to create custom actions beyond the basic movement actions
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def step(self, *args, **kwargs):
        """Implement the action step logic"""
        pass


class NoOpAction(BaseCustomAction):
    """Example custom action that does nothing (useful for testing)"""
    
    def step(self, *args, **kwargs):
        """Take no action, just return current observations"""
        return self._sim.step(None)
    
    @property
    def action_space(self):
        return EmptySpace()


class RandomTurnAction(BaseCustomAction):
    """Example custom action that performs a random turn (placeholder implementation)"""
    
    def step(self, *args, **kwargs):
        """Placeholder for random turn action - just returns current observations"""
        return self._sim.step(None)
    
    @property
    def action_space(self):
        return EmptySpace()


# For direct use (not inheritance) - utility functions using habitat-lab classes
def create_action_space_from_config(action_configs: Dict[str, Any]):
    """
    Utility function to create action space from configuration
    Uses habitat-lab's ActionSpace directly
    """
    from habitat.core.spaces import ActionSpace
    return ActionSpace(action_configs)


def create_measurements_from_config(measure_configs: Dict[str, Any]):
    """
    Utility function to create measurements from configuration
    Uses habitat-lab's Measurements directly
    """
    from habitat.core.embodied_task import Measurements
    from habitat.core.registry import registry
    
    measures = []
    for measure_name, measure_config in measure_configs.items():
        measure_type = registry.get_measure(measure_config.get('type'))
        measure = measure_type(**measure_config)
        measures.append(measure)
    
    return Measurements(measures)