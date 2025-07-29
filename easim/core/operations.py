"""
Core simulator operations and testing functionality
"""
import traceback
from easim.core.simulator import CoreSimulator, SimulatorConfig
from easim.utils.constants import TEST_SCENE_MP3D


def run_simulator_test(args):
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