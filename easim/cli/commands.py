"""Command execution handlers for EmbodiedAgentSim CLI"""
import traceback
from pathlib import Path

from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()

import habitat



def execute_command(args):
    """Execute the appropriate command based on arguments"""
    print("EmbodiedAgentSim - Habitat-based Simulation Framework")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        # Test command
        if args.command == 'test':
            print("Testing basic habitat-lab setup...")
            env = habitat.Env(
                config=habitat.get_config(f"benchmark/nav/{args.dataset.lower()}/pointnav_{args.dataset.lower()}_test.yaml")
            )
            print("Environment created successfully!")
            observations = env.reset()
            print("Initial observations:", observations)
            return 0

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Operation failed: {e}")
        traceback.print_exc()
        return 1

    return 0



