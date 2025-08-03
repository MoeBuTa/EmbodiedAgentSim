"""Command execution handlers for EmbodiedAgentSim CLI"""
import traceback
from pathlib import Path

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np

from easim.core.video_recorder import VideoRecorder
from easim.utils.constants import OUTPUT_DIR


def execute_command(args):
    """Execute the appropriate command based on arguments"""
    print("EmbodiedAgentSim - Habitat-based Simulation Framework")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        if args.command == "test":
            run_habitat_test(args)
        elif args.command == "record":
            run_recording_session(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Operation failed: {e}")
        traceback.print_exc()
        return 1

    return 0


def run_habitat_test(args):
    """Run basic habitat-lab test"""
    print(f"=== Habitat-Lab Test ({args.dataset}) ===")

    try:
        # Choose config based on dataset
        if args.dataset == "MP3D":
            config_path = "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
        else:  # HM3D
            config_path = "benchmark/nav/objectnav/objectnav_hm3d.yaml"

        # Create habitat environment
        env = habitat.Env(config=habitat.get_config(config_path))
        
        print("Environment created successfully!")
        print(f"Using config: {config_path}")

        # Test basic observations
        print("Testing observations...")
        observations = env.reset()
        
        for sensor_name, data in observations.items():
            print(f"  {sensor_name}: {data.shape} {data.dtype}")

        # Test actions
        print("Testing actions...")
        actions_to_test = [
            ("move_forward", HabitatSimActions.move_forward),
            ("turn_left", HabitatSimActions.turn_left), 
            ("turn_right", HabitatSimActions.turn_right)
        ]

        for action_name, action in actions_to_test:
            try:
                print(f"  Taking action: {action_name}")
                observations = env.step(action)
                print(f"    Success - got {len(observations)} sensor observations")
            except Exception as e:
                print(f"    Failed: {e}")

        print("Habitat-Lab test completed successfully!")
        env.close()

    except Exception as e:
        print(f"Habitat-Lab test failed: {e}")
        print("Make sure habitat-lab is properly installed and configured")
        traceback.print_exc()


def run_recording_session(args):
    """Run recording mode with habitat-lab"""
    print(f"=== Record Mode ({args.dataset}) ===")

    try:
        # Choose config based on dataset
        if args.dataset == "MP3D":
            config_path = "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
        else:  # HM3D
            config_path = "benchmark/nav/objectnav/objectnav_hm3d.yaml"

        # Create habitat environment
        env = habitat.Env(config=habitat.get_config(config_path))
        
        print("Environment created successfully!")
        print(f"Using config: {config_path}")

        # Setup output path  
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / args.video_name

        # Create video recorder
        recorder = VideoRecorder(str(video_path), fps=args.fps)
        recorder.start_recording()

        # Start episode
        observations = env.reset()

        # Simple random navigation
        print(f"Recording navigation for {args.max_steps} steps...")
        step_count = 0
        
        while not env.episode_over and step_count < args.max_steps:
            # Add frame to video
            if 'rgb' in observations:
                recorder.add_frame(observations['rgb'])

            # Random action with forward bias
            if np.random.random() < 0.7:
                action = HabitatSimActions.move_forward
            else:
                action = np.random.choice([HabitatSimActions.turn_left, HabitatSimActions.turn_right])

            # Take action
            observations = env.step(action)
            step_count += 1
            
            if step_count % 20 == 0:
                print(f"  Progress: {step_count}/{args.max_steps} steps")

        # Finalize recording
        recorder.stop_recording()

        # Save frames if requested
        if args.save_frames:
            frames_dir = output_dir / f"{video_path.stem}_frames"
            recorder.save_frames_as_images(str(frames_dir))

        print(f"Recording completed!")
        print(f"  Total steps: {step_count}")
        print(f"  Episode over: {env.episode_over}")
        print(f"  Video saved to: {video_path}")

        # Cleanup
        env.close()

    except Exception as e:
        print(f"Recording failed: {e}")
        print("Make sure habitat-lab is properly installed and configured")
        traceback.print_exc()