import argparse
import sys
from pathlib import Path

from easim.sim.simulator import SimulatorFactory
from easim.sim.video_recorder import SimulationRecorder, PygameVisualizer, RandomNavigationStrategy
from easim.utils.constants import PROJECT_DIR, DATA_PATH, OUTPUT_DIR, TEST_SCENE_MP3D


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

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive control')
    interactive_parser.add_argument('--dataset', choices=['MP3D', 'HM3D'], default='MP3D')
    interactive_parser.add_argument('--scene-path', help='Custom scene path')

    return parser


def run_simulator_mode(args):
    """Run basic simulator test"""
    print(f"=== Simulator Mode ({args.dataset}) ===")

    try:
        # Create simulator
        scene_path = args.scene_path if args.scene_path else None
        simulator = SimulatorFactory.create_simulator(args.dataset, scene_path)

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
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()


def run_record_mode(args):
    """Run recording mode"""
    print(f"=== Record Mode ({args.dataset}) ===")

    try:
        # Create simulator
        scene_path = args.scene_path if args.scene_path else None
        simulator = SimulatorFactory.create_simulator(args.dataset, scene_path)

        # Create recorder
        recorder = SimulationRecorder(simulator, args.output_dir)

        # Create navigation strategy
        strategy = RandomNavigationStrategy(
            max_steps=args.max_steps,
            forward_prob=0.7
        )

        # Record navigation
        print(f"Recording navigation for {args.max_steps} steps...")
        result = recorder.record_navigation(
            strategy,
            video_filename=args.video_name,
            save_frames=True
        )

        print(f"Recording completed!")
        print(f"  Total steps: {result['total_steps']}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Video saved to: {result['video_path']}")

        # Cleanup
        simulator.close()

    except Exception as e:
        print(f"Recording failed: {e}")
        print("Make sure the scene files exist and cv2 is installed")


def run_interactive_mode(args):
    """Run interactive mode with pygame"""
    print(f"=== Interactive Mode ({args.dataset}) ===")
    print("Controls: W/↑=Forward, A/←=Left, D/→=Right, ESC=Quit")

    try:
        # Create simulator
        scene_path = args.scene_path if args.scene_path else None
        simulator = SimulatorFactory.create_simulator(args.dataset, scene_path)

        # Create visualizer
        visualizer = PygameVisualizer(window_size=(800, 600))

        # Run interactive simulation
        print("Starting interactive simulation...")
        visualizer.run_interactive_simulation(simulator)

        # Cleanup
        simulator.close()

    except ImportError:
        print("pygame is required for interactive mode")
        print("Install with: pip install pygame")
    except Exception as e:
        print(f"Interactive mode failed: {e}")


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
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())