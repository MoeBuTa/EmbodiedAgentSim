"""Command execution handlers for EmbodiedAgentSim CLI"""
import traceback

from easim.agents.sample import SampleAgent
from easim.benchmark.benchmark import HabitatBenchmark
from easim.utils.constants import BENCHMARK_CONFIG

from easim.demo.interactive import interactive


def execute_command(args):
    """Execute the appropriate command based on arguments"""
    print("EmbodiedAgentSim - Habitat-based Simulation Framework")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        if args.command == 'interactive':
            interactive()
        elif args.command == 'list-tasks':
            print("Available benchmark tasks:")
            for task in BENCHMARK_CONFIG:
                print(f" - {task}")
            return 0
        elif args.command == 'benchmark':
            return benchmark(args)
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


def benchmark(args):
    """Run benchmark evaluation"""
    print(f"Task: {args.task}")
    print(f"Episodes: {args.episodes if args.episodes is not None else 'all'}")
    print(f"Agent: {args.agent}")
    print(f"Stage: {args.stage}")
    print(f"episode count: {args.episodes if args.episodes is not None else 'all'}")
    print(f"Record video: {args.record}")
    print("-" * 30)

    # Initialize benchmark
    try:
        habitat_benchmark = HabitatBenchmark(task_name=args.task, agent=args.agent, stage=args.stage,
                                             episodes=args.episodes)
    except KeyError:
        print(f"Error: Task '{args.task}' not found in benchmark configurations")
        return 1

    # Run evaluation
    print(f"Starting evaluation...")
    habitat_benchmark.evaluate(
        enable_record=args.record
    )

    return 0
