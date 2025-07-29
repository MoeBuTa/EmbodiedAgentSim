import argparse
import sys
import time
import traceback
import habitat_sim
import numpy as np
from pathlib import Path
from easim.core.simulator import CoreSimulator, SimulatorConfig
from easim.core.video_recorder import VideoRecorder, RandomNavigationStrategy, PygameVisualizer
from easim.utils.constants import (
    PROJECT_DIR, DATA_PATH, OUTPUT_DIR, TEST_SCENE_MP3D, TEST_SCENE_HM3D,
    DEFAULT_SENSOR_RESOLUTION, DEFAULT_SENSOR_HEIGHT, DEFAULT_FORWARD_STEP, DEFAULT_TURN_ANGLE
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="EmbodiedAgentSim - Habitat-based simulation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulator test
  easim simulator --dataset MP3D

  # Record random navigation video
  easim record --dataset HM3D --output-dir videos

  # Interactive control
  easim interactive --dataset MP3D
        """
    )

    # Add subcommands instead of mode
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Simulator command
    sim_parser = subparsers.add_parser('simulator', help='Test basic simulator')
    sim_parser.add_argument('--dataset', choices=['MP3D', 'HM3D'], default='MP3D')
    sim_parser.add_argument('--scene-path', help='Custom scene path')

    # Record command
    record_parser = subparsers.add_parser('record', help='Record navigation video')
    record_parser.add_argument('--dataset', choices=['MP3D', 'HM3D'], default='MP3D')
    record_parser.add_argument('--scene-path', help='Custom scene path')
    record_parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Output directory')
    record_parser.add_argument('--video-name', default='simulation.mp4', help='Video filename')
    record_parser.add_argument('--max-steps', type=int, default=100, help='Max navigation steps')
    record_parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    record_parser.add_argument('--save-frames', action='store_true', help='Save individual frames')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive control')
    interactive_parser.add_argument('--dataset', choices=['MP3D', 'HM3D'], default='MP3D')
    interactive_parser.add_argument('--scene-path', help='Custom scene path')

    return parser


def run_simulator_mode(args):
    """Run basic simulator test"""
    print(f"=== Simulator Mode ({args.dataset}) ===")

    try:
        # Create simulator using new core module
        config = SimulatorConfig(
            scene_path=args.scene_path,
            dataset_type=args.dataset
        )
        simulator = CoreSimulator(config)

        print(f"Simulator created successfully!")
        print(f"Scene info: {simulator.get_scene_info()}")

        # Test basic functionality
        print("Testing basic observations...")
        observations = simulator.get_observations()

        for sensor_name, data in observations.items():
            print(f"  {sensor_name}: {data.shape} {data.dtype}")

        # Test a few actions with error handling
        print("Testing actions...")
        actions_to_test = ["move_forward", "turn_left", "turn_right"]

        for action in actions_to_test:
            try:
                print(f"  Taking action: {action}")
                observations = simulator.step(action)
                print(f"    Success - got {len(observations)} sensor observations")
            except Exception as e:
                print(f"    Failed: {e}")
                continue

        print("Simulator test completed successfully!")

        # Cleanup
        simulator.close()

    except Exception as e:
        print(f"Simulator test failed: {e}")
        print("Make sure the scene files exist in the data directory")
        print(f"Expected MP3D scene at: {TEST_SCENE_MP3D}")

        # Print additional debug info
        print("\nFull error traceback:")
        traceback.print_exc()


def run_record_mode(args):
    """Run recording mode with error handling"""
    print(f"=== Record Mode ({args.dataset}) ===")

    try:
        # Create simulator using simplified approach

        # Determine scene path
        if args.scene_path:
            scene_path = args.scene_path
        elif args.dataset == "MP3D":
            scene_path = str(TEST_SCENE_MP3D)
        else:
            scene_path = str(TEST_SCENE_HM3D)

        # Create basic simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.gpu_device_id = 0

        # Create basic agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.2

        # Create RGB sensor
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [DEFAULT_SENSOR_RESOLUTION[1], DEFAULT_SENSOR_RESOLUTION[0]]
        rgb_sensor.position = [0.0, DEFAULT_SENSOR_HEIGHT, 0.0]
        rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        agent_cfg.sensor_specifications = [rgb_sensor]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=DEFAULT_FORWARD_STEP)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=DEFAULT_TURN_ANGLE)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=DEFAULT_TURN_ANGLE)
            )
        }

        # Create and initialize simulator
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(sim_cfg)

        print("Simulator created successfully!")

        # Setup output path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / args.video_name

        # Create video recorder
        recorder = VideoRecorder(str(video_path), fps=args.fps)
        recorder.start_recording()

        # Create navigation strategy
        strategy = RandomNavigationStrategy(max_steps=args.max_steps, forward_prob=0.7)

        # Record navigation
        print(f"Recording navigation for {args.max_steps} steps...")
        step_count = 0
        start_time = time.time()

        while not strategy.is_done(sim.get_sensor_observations(), step_count):
            # Get observations
            observations = sim.get_sensor_observations()

            # Add frame to video
            if 'rgb' in observations:
                recorder.add_frame(observations['rgb'])

            # Get next action
            action = strategy.get_next_action(observations, step_count)

            # Take action
            sim.step(action)
            step_count += 1

        # Finalize recording
        end_time = time.time()
        recorder.stop_recording()

        # Save frames if requested
        if args.save_frames:
            frames_dir = output_dir / f"{video_path.stem}_frames"
            recorder.save_frames_as_images(str(frames_dir))

        print(f"Recording completed!")
        print(f"  Total steps: {step_count}")
        print(f"  Duration: {end_time - start_time:.2f}s")
        print(f"  Video saved to: {video_path}")

        # Cleanup
        sim.close()

    except Exception as e:
        print(f"Recording failed: {e}")
        print("Make sure the scene files exist and cv2 is installed")
        traceback.print_exc()

def run_interactive_mode(args):
    """Run interactive mode with pygame"""
    print(f"=== Interactive Mode ({args.dataset}) ===")
    print("Controls: W/↑=Forward, A/←=Left, D/→=Right, ESC=Quit")

    try:
        # Create simulator using simplified approach (same as record mode)

        # Determine scene path
        if args.scene_path:
            scene_path = args.scene_path
        elif args.dataset == "MP3D":
            scene_path = str(TEST_SCENE_MP3D)
        else:
            scene_path = str(TEST_SCENE_HM3D)

        # Create basic simulator (same setup as record mode)
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.gpu_device_id = 0

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.2

        # Create RGB sensor for visualization
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [DEFAULT_SENSOR_RESOLUTION[1], DEFAULT_SENSOR_RESOLUTION[0]]
        rgb_sensor.position = [0.0, DEFAULT_SENSOR_HEIGHT, 0.0]
        rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        agent_cfg.sensor_specifications = [rgb_sensor]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=DEFAULT_FORWARD_STEP)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=DEFAULT_TURN_ANGLE)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=DEFAULT_TURN_ANGLE)
            )
        }

        # Create simple simulator wrapper for pygame
        class SimpleSimWrapper:
            def __init__(self, habitat_sim):
                self.sim = habitat_sim

            def get_observations(self):
                return self.sim.get_sensor_observations()

            def step(self, action):
                return self.sim.step(action)

            def reset(self):
                return self.sim.reset()

        # Initialize and run
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(sim_cfg)
        wrapper = SimpleSimWrapper(sim)

        print("Starting interactive simulation...")
        visualizer = PygameVisualizer(window_size=(800, 600))
        visualizer.run_interactive_simulation(wrapper)

        sim.close()

    except ImportError:
        print("pygame is required for interactive mode")
        print("Install with: pip install pygame")
    except Exception as e:
        print(f"Interactive mode failed: {e}")
        traceback.print_exc()


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 1

    print("EmbodiedAgentSim - Habitat-based Simulation Framework")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Data directory: {DATA_PATH}")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        if args.command == "simulator":
            run_simulator_mode(args)
        elif args.command == "record":
            run_record_mode(args)
        elif args.command == "interactive":
            run_interactive_mode(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Operation failed: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())