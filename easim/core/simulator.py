"""
Core simulator for embodied AI with integrated video recording
"""
import habitat_sim
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass

from easim.core.agents import BaseAgent, AgentFactory, get_task_agent
from easim.core.video_recorder import (
    BaseNavigationStrategy, RandomNavigationStrategy,
    FixedPathStrategy, record_navigation, run_interactive_simulation
)
from easim.utils.constants import (
    DATA_PATH, TEST_SCENE_MP3D, TEST_SCENE_HM3D,
    MP3D_SCENE_DATASET, HM3D_SCENE_DATASET
)


@dataclass
class SimulatorConfig:
    """Simulator configuration"""
    scene_path: Optional[str] = None
    scene_dataset: Optional[str] = None
    dataset_type: str = "MP3D"  # MP3D, HM3D, or custom
    gpu_device_id: int = 0
    enable_physics: bool = False
    random_seed: Optional[int] = None
    allow_sliding: bool = True
    frustum_culling: bool = True


class CoreSimulator:
    """Core simulator for embodied AI tasks with video recording support"""

    def __init__(self,
                 config: SimulatorConfig,
                 agent: Optional[BaseAgent] = None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.sim = None
        self.agents = {}
        self.current_scene_id = None

        # Create default agent if none provided
        if agent is None and agent_config is None:
            from easim.core.agents import NavigationAgent
            agent = NavigationAgent()

        # Initialize simulator with agent
        self._initialize_simulator(agent or AgentFactory.create_agent(**agent_config))

    def _initialize_simulator(self, agent: BaseAgent):
        """Initialize the habitat simulator backend with an agent"""
        sim_cfg = self._create_simulator_config(agent)
        self.sim = habitat_sim.Simulator(sim_cfg)

        # Store the initial agent
        self.agents[0] = {
            'agent': agent,
            'habitat_agent': self.sim.agents[0]
        }

    def _create_simulator_config(self, agent: BaseAgent) -> habitat_sim.Configuration:
        """Create habitat simulator configuration with agent"""
        # Backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.gpu_device_id = self.config.gpu_device_id
        backend_cfg.enable_physics = self.config.enable_physics
        backend_cfg.allow_sliding = self.config.allow_sliding
        backend_cfg.frustum_culling = self.config.frustum_culling

        # Scene configuration
        backend_cfg.scene_id = self._resolve_scene_path()
        backend_cfg.scene_dataset_config_file = self._resolve_scene_dataset()

        # Random seed
        if self.config.random_seed is not None:
            backend_cfg.random_seed = self.config.random_seed

        # Get agent configuration
        agent_cfg = agent.get_habitat_agent_config()

        # Create configuration with agent
        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def _resolve_scene_path(self) -> str:
        """Resolve scene path based on configuration"""
        if self.config.scene_path:
            return self.config.scene_path

        # Use default scenes based on dataset type
        dataset_type = self.config.dataset_type.upper()
        if dataset_type == "MP3D":
            return str(TEST_SCENE_MP3D)
        elif dataset_type == "HM3D":
            return str(TEST_SCENE_HM3D)
        else:
            # Fallback to MP3D
            return str(TEST_SCENE_MP3D)

    def _resolve_scene_dataset(self) -> str:
        """Resolve scene dataset configuration"""
        if self.config.scene_dataset:
            return self.config.scene_dataset

        # Use default dataset configs
        dataset_type = self.config.dataset_type.upper()
        if dataset_type == "MP3D" and MP3D_SCENE_DATASET.exists():
            return str(MP3D_SCENE_DATASET)
        elif dataset_type == "HM3D" and HM3D_SCENE_DATASET.exists():
            return str(HM3D_SCENE_DATASET)
        else:
            # Return empty string for basic scene loading
            return ""

    def add_agent(self, agent_id: int, agent: BaseAgent):
        """Add an agent to the simulator"""
        if self.sim is None:
            raise RuntimeError("Simulator not initialized")

        # Get habitat agent configuration
        agent_cfg = agent.get_habitat_agent_config()

        # Initialize agent in simulator
        habitat_agent = self.sim.initialize_agent(agent_id, agent_cfg)

        # Store agent reference
        self.agents[agent_id] = {
            'agent': agent,
            'habitat_agent': habitat_agent
        }

    def get_agent(self, agent_id: int = 0) -> Optional[BaseAgent]:
        """Get agent by ID"""
        if agent_id in self.agents:
            return self.agents[agent_id]['agent']
        return None

    def get_observations(self, agent_id: int = 0) -> Dict[str, np.ndarray]:
        """Get current observations for agent"""
        if self.sim is None:
            raise RuntimeError("Simulator not initialized")
        return self.sim.get_sensor_observations(agent_id)

    def step(self, action: Union[str, int], agent_id: int = 0) -> Dict[str, np.ndarray]:
        """Take an action and return observations"""
        if self.sim is None:
            raise RuntimeError("Simulator not initialized")

        # Convert action index to name if needed
        if isinstance(action, int):
            agent = self.get_agent(agent_id)
            if agent:
                action_names = agent.get_action_names()
                if 0 <= action < len(action_names):
                    action = action_names[action]
                else:
                    raise ValueError(f"Action index {action} out of range")

        return self.sim.step(action, agent_id)

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        """Get agent state"""
        if self.sim is None or agent_id not in self.agents:
            raise RuntimeError("Agent not found")
        return self.agents[agent_id]['habitat_agent'].get_state()

    def set_agent_state(self,
                        position: List[float],
                        rotation: Optional[List[float]] = None,
                        agent_id: int = 0):
        """Set agent state"""
        if self.sim is None or agent_id not in self.agents:
            raise RuntimeError("Agent not found")

        state = habitat_sim.AgentState()
        state.position = np.array(position, dtype=np.float32)

        # if rotation is not None:
        #     state.rotation = np.quaternion(*rotation)

        self.agents[agent_id]['habitat_agent'].set_state(state)

    def reset(self):
        """Reset the simulator"""
        if self.sim is not None:
            self.sim.reset()

    def reconfigure(self, config: SimulatorConfig):
        """Reconfigure simulator with new settings"""
        self.config = config

        # Get the current primary agent to maintain it
        primary_agent = self.agents.get(0, {}).get('agent')
        if primary_agent is None:
            from easim.core.agents import NavigationAgent
            primary_agent = NavigationAgent()

        self.close()
        self._initialize_simulator(primary_agent)

        # Re-add any additional agents
        agents_to_readd = [(aid, info['agent']) for aid, info in self.agents.items() if aid != 0]

        for agent_id, agent in agents_to_readd:
            self.add_agent(agent_id, agent)

    def load_scene(self, scene_path: str, scene_dataset: Optional[str] = None):
        """Load a new scene"""
        new_config = SimulatorConfig(
            scene_path=scene_path,
            scene_dataset=scene_dataset or self.config.scene_dataset,
            dataset_type=self.config.dataset_type,
            gpu_device_id=self.config.gpu_device_id,
            enable_physics=self.config.enable_physics,
            random_seed=self.config.random_seed
        )
        self.reconfigure(new_config)
        self.current_scene_id = scene_path

    def get_scene_info(self) -> Dict[str, Any]:
        """Get scene information"""
        scene_path = self._resolve_scene_path()
        return {
            "scene_path": scene_path,
            "scene_dataset": self._resolve_scene_dataset(),
            "dataset_type": self.config.dataset_type,
            "scene_name": Path(scene_path).stem,
            "num_agents": len(self.agents),
            "physics_enabled": self.config.enable_physics
        }

    def get_pathfinder(self):
        """Get pathfinder for navigation"""
        if self.sim is None:
            return None
        return self.sim.pathfinder

    def is_navigable(self, position: List[float]) -> bool:
        """Check if position is navigable"""
        pathfinder = self.get_pathfinder()
        if pathfinder is None:
            return True  # Assume navigable if no pathfinder
        return pathfinder.is_navigable(position)

    def get_random_navigable_point(self) -> Optional[np.ndarray]:
        """Get random navigable point"""
        pathfinder = self.get_pathfinder()
        if pathfinder is None:
            return None
        return pathfinder.get_random_navigable_point()

    def find_path(self, start: List[float], end: List[float]) -> List[List[float]]:
        """Find path between two points"""
        pathfinder = self.get_pathfinder()
        if pathfinder is None:
            return [start, end]  # Direct path if no pathfinder

        path = pathfinder.find_path(
            habitat_sim.ShortestPath()
        )
        path.requested_start = start
        path.requested_end = end

        found_path = pathfinder.find_path(path)
        return found_path.points if found_path.points else [start, end]

    # Video recording methods
    def record_navigation(self,
                         strategy: BaseNavigationStrategy,
                         output_path: str,
                         fps: int = 30,
                         save_frames: bool = False) -> Dict[str, Any]:
        """Record navigation video using strategy"""
        return record_navigation(
            simulator=self,
            strategy=strategy,
            output_path=output_path,
            fps=fps,
            save_frames=save_frames
        )

    def record_random_navigation(self,
                                output_path: str,
                                max_steps: int = 100,
                                forward_prob: float = 0.7,
                                fps: int = 30,
                                save_frames: bool = False) -> Dict[str, Any]:
        """Record random navigation video"""
        strategy = RandomNavigationStrategy(max_steps=max_steps, forward_prob=forward_prob)
        return self.record_navigation(strategy, output_path, fps, save_frames)

    def record_fixed_path(self,
                         action_sequence: List[str],
                         output_path: str,
                         fps: int = 30,
                         save_frames: bool = False) -> Dict[str, Any]:
        """Record navigation with fixed action sequence"""
        strategy = FixedPathStrategy(action_sequence)
        return self.record_navigation(strategy, output_path, fps, save_frames)

    def run_interactive(self):
        """Run interactive simulation with pygame controls"""
        run_interactive_simulation(self)

    def close(self):
        """Close the simulator"""
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.agents.clear()


class TaskSimulator(CoreSimulator):
    """Simulator configured for specific tasks"""

    def __init__(self,
                 task_type: str,
                 dataset_type: str = "MP3D",
                 scene_path: Optional[str] = None,
                 **kwargs):
        # Create appropriate agent for task
        agent = get_task_agent(task_type, **kwargs)

        # Create simulator config
        config = SimulatorConfig(
            scene_path=scene_path,
            dataset_type=dataset_type,
            **{k: v for k, v in kwargs.items()
               if k in ['gpu_device_id', 'enable_physics', 'random_seed']}
        )

        super().__init__(config, agent)
        self.task_type = task_type

    def get_scene_info(self) -> Dict[str, Any]:
        """Get task-specific scene information"""
        info = super().get_scene_info()
        info["task_type"] = self.task_type
        return info


class SimulatorFactory:
    """Factory for creating simulators"""

    @staticmethod
    def create_simulator(simulator_type: str = "core", **kwargs) -> CoreSimulator:
        """Create simulator by type"""
        simulator_type = simulator_type.lower()

        if simulator_type == "core":
            config = SimulatorConfig(**kwargs)
            return CoreSimulator(config)
        elif simulator_type == "task":
            return TaskSimulator(**kwargs)
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")

    @staticmethod
    def create_task_simulator(task_type: str,
                              dataset_type: str = "MP3D",
                              **kwargs) -> TaskSimulator:
        """Create task-specific simulator"""
        return TaskSimulator(task_type, dataset_type, **kwargs)


# Convenience functions for backward compatibility
def create_mp3d_simulator(scene_path: Optional[str] = None, **kwargs) -> CoreSimulator:
    """Create MP3D simulator"""
    config = SimulatorConfig(
        scene_path=scene_path,
        dataset_type="MP3D",
        **kwargs
    )
    return CoreSimulator(config)


def create_hm3d_simulator(scene_path: Optional[str] = None, **kwargs) -> CoreSimulator:
    """Create HM3D simulator"""
    config = SimulatorConfig(
        scene_path=scene_path,
        dataset_type="HM3D",
        **kwargs
    )
    return CoreSimulator(config)