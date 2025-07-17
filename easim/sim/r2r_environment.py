import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from easim.sim.simulator import SimulatorFactory, SimulatorConfig
from easim.datasets.r2r_loader import R2RDatasetLoader, R2REpisode
from easim.utils.constants import DATA_PATH


class R2REnvironment:
    """Environment for R2R Vision-and-Language Navigation"""

    def __init__(self, split: str = "val_seen"):
        self.split = split
        self.loader = R2RDatasetLoader()
        self.episodes = self.loader.load_split(split)
        self.current_episode_idx = 0
        self.current_episode = None
        self.simulator = None
        self.current_instruction_idx = 0

        print(f"✅ R2R Environment loaded with {len(self.episodes)} episodes from {split}")

    def reset(self, episode_idx: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment to a new episode"""
        if episode_idx is not None:
            self.current_episode_idx = episode_idx

        # Get current episode
        if self.current_episode_idx >= len(self.episodes):
            self.current_episode_idx = 0

        self.current_episode = self.episodes[self.current_episode_idx]
        self.current_instruction_idx = 0

        # Create simulator for this scene
        scene_path = self._get_scene_path(self.current_episode.scan)

        if scene_path and scene_path.exists():
            # Close existing simulator
            if self.simulator:
                self.simulator.close()

            # Create new simulator
            config = SimulatorConfig(
                scene_path=str(scene_path),
                scene_dataset="",  # Basic setup
                width=640,
                height=480
            )

            from easim.sim.simulator import BaseSimulator
            class R2RSimulator(BaseSimulator):
                def get_scene_info(self):
                    return {"dataset": "R2R", "scene": self.config.scene_path}

            self.simulator = R2RSimulator(config)

            # Set agent to starting position
            start_pos = self.current_episode.path[0]  # First position in path
            self.simulator.set_agent_state(start_pos, [1, 0, 0, 0])  # Default rotation

        else:
            print(f"⚠️ Scene not found: {scene_path}")
            self.simulator = None

        return self.get_current_state()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state including observations and instruction"""
        state = {
            "episode_id": self.current_episode.episode_id,
            "scan": self.current_episode.scan,
            "instruction": self.get_current_instruction(),
            "target_position": self.current_episode.path[-1],  # Last position in path
            "step": 0,
            "observations": None
        }

        if self.simulator:
            try:
                state["observations"] = self.simulator.get_observations()
                state["agent_state"] = self.simulator.get_agent_state()
            except Exception as e:
                print(f"Error getting observations: {e}")

        return state

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take an action and return new state"""
        if not self.simulator:
            return self.get_current_state(), 0.0, True, {"error": "No simulator"}

        try:
            # Take action
            observations = self.simulator.step(action)

            # Calculate reward (distance to target)
            reward = self._calculate_reward()

            # Check if done (reached target or max steps)
            done = self._is_done()

            # Get info
            info = {
                "success": self._check_success(),
                "distance_to_target": self._distance_to_target(),
                "spl": self._calculate_spl()
            }

            # Update state
            state = self.get_current_state()
            state["observations"] = observations

            return state, reward, done, info

        except Exception as e:
            print(f"Error in step: {e}")
            return self.get_current_state(), 0.0, True, {"error": str(e)}

    def get_current_instruction(self) -> str:
        """Get current instruction for the episode"""
        if not self.current_episode:
            return ""

        instructions = self.current_episode.instructions
        if self.current_instruction_idx < len(instructions):
            return instructions[self.current_instruction_idx]
        return instructions[0] if instructions else ""

    def next_instruction(self):
        """Move to next instruction variant"""
        if self.current_episode:
            max_idx = len(self.current_episode.instructions) - 1
            self.current_instruction_idx = min(self.current_instruction_idx + 1, max_idx)

    def next_episode(self):
        """Move to next episode"""
        self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episodes)
        return self.reset()

    def _get_scene_path(self, scan: str) -> Optional[Path]:
        """Get path to MP3D scene file"""
        # Try different possible locations
        possible_paths = [
            DATA_PATH / "scene_datasets" / "mp3d" / scan / f"{scan}.glb",
            DATA_PATH / "scene_datasets" / "mp3d" / scan / f"{scan}.house",
            DATA_PATH / "scene_datasets" / "mp3d_example" / scan / f"{scan}.glb",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _calculate_reward(self) -> float:
        """Calculate reward for current position"""
        if not self.simulator or not self.current_episode:
            return 0.0

        # Distance-based reward
        current_pos = self.simulator.get_agent_state().position
        target_pos = np.array(self.current_episode.path[-1])
        distance = np.linalg.norm(current_pos - target_pos)

        # Negative distance as reward (closer = higher reward)
        return -distance

    def _distance_to_target(self) -> float:
        """Calculate distance to target"""
        if not self.simulator or not self.current_episode:
            return float('inf')

        current_pos = self.simulator.get_agent_state().position
        target_pos = np.array(self.current_episode.path[-1])
        return np.linalg.norm(current_pos - target_pos)

    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Simple version: check if close to target
        return self._distance_to_target() < 3.0  # 3 meter threshold

    def _check_success(self) -> bool:
        """Check if navigation was successful"""
        return self._distance_to_target() < 3.0

    def _calculate_spl(self) -> float:
        """Calculate Success weighted by Path Length (SPL)"""
        if not self._check_success():
            return 0.0

        # Simplified SPL calculation
        optimal_length = self.current_episode.distance
        # TODO: Calculate actual path length taken
        actual_length = optimal_length  # Placeholder

        return optimal_length / max(actual_length, optimal_length)

    def print_episode_info(self):
        """Print information about current episode"""
        if not self.current_episode:
            print("No episode loaded")
            return

        ep = self.current_episode
        print(f"\n=== Episode {ep.episode_id} ===")
        print(f"Scene: {ep.scan}")
        print(f"Instructions:")
        for i, instruction in enumerate(ep.instructions):
            marker = "→" if i == self.current_instruction_idx else " "
            print(f"  {marker} {instruction}")
        print(f"Path length: {ep.distance:.1f}m")
        print(f"Path points: {len(ep.path)}")

        if self.simulator:
            try:
                agent_state = self.simulator.get_agent_state()
                target_pos = np.array(ep.path[-1])
                distance = np.linalg.norm(agent_state.position - target_pos)
                print(f"Current distance to target: {distance:.1f}m")
            except:
                pass


def demo_r2r_usage():
    """Demonstration of R2R environment usage"""
    print("=== R2R Environment Demo ===")

    # Create environment
    env = R2REnvironment(split="val_seen")

    # Reset to first episode
    state = env.reset(episode_idx=0)
    env.print_episode_info()

    # Take some actions
    actions = ["move_forward", "turn_left", "move_forward", "turn_right", "move_forward"]

    for i, action in enumerate(actions):
        print(f"\nStep {i + 1}: Taking action '{action}'")
        state, reward, done, info = env.step(action)

        print(f"Reward: {reward:.2f}")
        print(f"Distance to target: {info['distance_to_target']:.1f}m")
        print(f"Success: {info['success']}")

        if done:
            print("Episode finished!")
            break

    # Close environment
    if env.simulator:
        env.simulator.close()


if __name__ == "__main__":
    demo_r2r_usage()