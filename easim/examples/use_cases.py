#!/usr/bin/env python3
"""
Simple use case examples for EmbodiedAgentSim
"""

import numpy as np
from pathlib import Path

from easim.sim.simulator import SimulatorFactory
from easim.sim.video_recorder import (
    SimulationRecorder, PygameVisualizer,
    RandomNavigationStrategy, FixedPathStrategy
)
from easim.utils.constants import (
    OUTPUT_DIR, ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT
)


def random_exploration_demo(dataset: str = "MP3D", output_dir: str = None):
    """Demo: Random exploration with video recording"""
    print(f"=== Random Exploration Demo ({dataset}) ===")

    if output_dir is None:
        output_dir = OUTPUT_DIR / "demos"

    try:
        # Create simulator
        simulator = SimulatorFactory.create_simulator(dataset)
        print(f"Created {dataset} simulator")
        print(f"Scene info: {simulator.get_scene_info()}")

        # Create recorder
        recorder = SimulationRecorder(simulator, output_dir)

        # Create random navigation strategy
        strategy = RandomNavigationStrategy(max_steps=100, forward_prob=0.7)

        # Record navigation
        print("Recording random navigation...")
        result = recorder.record_navigation(
            strategy,
            video_filename=f"random_exploration_{dataset.lower()}.mp4",
            save_frames=True
        )

        print(f"Recording complete!")
        print(f"Total steps: {result['total_steps']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Video saved to: {result['video_path']}")

        # Cleanup
        simulator.close()
        return result

    except Exception as e:
        print(f"Random exploration demo failed: {e}")
        return None


def fixed_path_demo(dataset: str = "HM3D", output_dir: str = None):
    """Demo: Fixed action sequence navigation"""
    print(f"=== Fixed Path Demo ({dataset}) ===")

    if output_dir is None:
        output_dir = OUTPUT_DIR / "demos"

    try:
        # Create simulator
        simulator = SimulatorFactory.create_simulator(dataset)

        # Define action sequence: forward, turn, forward, turn, etc.
        action_sequence = (
                [ACTION_MOVE_FORWARD] * 10 +
                [ACTION_TURN_RIGHT] * 3 +
                [ACTION_MOVE_FORWARD] * 10 +
                [ACTION_TURN_LEFT] * 3 +
                [ACTION_MOVE_FORWARD] * 10
        )

        # Create recorder and strategy
        recorder = SimulationRecorder(simulator, output_dir)
        strategy = FixedPathStrategy(action_sequence)

        # Record navigation
        print("Recording fixed path navigation...")
        result = recorder.record_navigation(
            strategy,
            video_filename=f"fixed_path_{dataset.lower()}.mp4"
        )

        print(f"Recording complete!")
        print(f"Total steps: {result['total_steps']}")
        print(f"Video saved to: {result['video_path']}")

        # Cleanup
        simulator.close()
        return result

    except Exception as e:
        print(f"Fixed path demo failed: {e}")
        return None


def interactive_pygame_demo(dataset: str = "MP3D"):
    """Demo: Interactive control with pygame"""
    print(f"=== Interactive Pygame Demo ({dataset}) ===")
    print("Use W/A/D or arrow keys to control the agent")
    print("Press ESC to quit")

    try:
        # Create simulator
        simulator = SimulatorFactory.create_simulator(dataset)

        # Create visualizer
        visualizer = PygameVisualizer(window_size=(800, 600))

        # Run interactive simulation
        print("Starting interactive simulation...")
        visualizer.run_interactive_simulation(simulator)

        # Cleanup
        simulator.close()

    except ImportError:
        print("pygame is required for interactive demo")
        print("Install with: pip install pygame")
    except Exception as e:
        print(f"Interactive demo failed: {e}")


def comparison_demo():
    """Demo: Compare performance across different setups"""
    print("=== Comparison Demo ===")

    results = {}

    # Test different simulators
    for dataset in ["MP3D", "HM3D"]:
        print(f"\nTesting {dataset} random exploration...")
        try:
            result = random_exploration_demo(dataset, OUTPUT_DIR / f"{dataset.lower()}_test")
            if result:
                results[f"{dataset}_exploration"] = {
                    "steps": result["total_steps"],
                    "duration": result["duration"]
                }
        except Exception as e:
            print(f"{dataset} test failed: {e}")
            results[f"{dataset}_exploration"] = None

    # Print comparison
    print("\n=== Final Comparison ===")
    for test_name, result in results.items():
        if result:
            print(f"{test_name}: {result}")
        else:
            print(f"{test_name}: Failed")

    return results


def main():
    """Run all simple demos"""
    print("Running EmbodiedAgentSim Simple Demos")
    print("-" * 40)

    try:
        # Run basic demos
        random_exploration_demo("MP3D")
        print()

        fixed_path_demo("MP3D")  # Use MP3D since it's more likely to work
        print()

        # Run comparison
        comparison_demo()

    except KeyboardInterrupt:
        print("\nDemos interrupted by user")
    except Exception as e:
        print(f"Demos failed: {e}")


if __name__ == "__main__":
    main()