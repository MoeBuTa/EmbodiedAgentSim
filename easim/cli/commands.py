"""
Command execution handlers for EmbodiedAgentSim CLI
"""
import traceback
from easim.core.operations import run_simulator_test
from easim.core.video_recorder import run_recording_session, run_interactive_session
from easim.utils.constants import PROJECT_DIR, DATA_PATH


def execute_command(args):
    """Execute the appropriate command based on arguments"""
    print("EmbodiedAgentSim - Habitat-based Simulation Framework")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Data directory: {DATA_PATH}")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        if args.command == "simulator":
            run_simulator_test(args)
        elif args.command == "record":
            run_recording_session(args)
        elif args.command == "interactive":
            run_interactive_session(args)
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