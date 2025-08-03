"""Example demonstrating video recording with habitat-lab"""
from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np
from pathlib import Path

from easim.core.video_recorder import VideoRecorder
from easim.utils.constants import OUTPUT_DIR


def main():
    """Basic example of video recording with habitat-lab"""
    
    # Create habitat environment
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    )
    
    print("Environment creation successful")

    # Setup video recording
    video_path = OUTPUT_DIR / "example_recording.mp4"

    recorder = VideoRecorder(str(video_path), fps=30)
    recorder.start_recording()

    # Start episode
    observations = env.reset()
    print("Episode started")

    # Navigation loop with video recording
    max_steps = 50
    step_count = 0
    
    while not env.episode_over and step_count < max_steps:
        # Record frame
        if 'rgb' in observations:
            recorder.add_frame(observations['rgb'])
        
        # Simple random navigation
        if np.random.random() < 0.7:
            action = HabitatSimActions.move_forward
        else:
            action = np.random.choice([HabitatSimActions.turn_left, HabitatSimActions.turn_right])
        
        # Take action
        observations = env.step(action)
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"Step {step_count}/{max_steps}")

    # Finalize recording
    recorder.stop_recording()
    print(f"Video saved to: {video_path}")
    print(f"Episode finished after {step_count} steps")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()