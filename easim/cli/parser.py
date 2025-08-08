"""
Command line argument parser for EmbodiedAgentSim
"""
import argparse
from easim.utils.constants import BENCHMARK_CONFIG


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="EmbodiedAgentSim - Habitat-based simulation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive navigation demo
  easim interactive

  # List all available benchmark tasks
  easim list-tasks

  # Benchmark agent on ObjectNav HM3D
  easim benchmark --task objectnav_hm3d --episodes 10

  # Benchmark with video recording
  easim benchmark --task pointnav_mp3d --episodes 5 --video

  # Use different agent (when available)
  easim benchmark --task imagenav_mp3d --agent llm --episodes 3
        """
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run interactive navigation demo')

    # List tasks command
    list_tasks_parser = subparsers.add_parser('list-tasks', help='List available benchmark tasks')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run agent benchmarking')
    benchmark_parser.add_argument(
        '--task', 
        choices=list(BENCHMARK_CONFIG.keys()), 
        default='objectnav_hm3d',
        help='Benchmark task to run'
    )
    benchmark_parser.add_argument(
        '--episodes', 
        type=int, 
        default=None, 
        help='Number of episodes to run (default: run all available episodes)'
    )
    benchmark_parser.add_argument(
        '--record',
        action='store_true', 
        help='Record videos, images of episodes'
    )
    benchmark_parser.add_argument(
        '--agent', 
        default='sample', 
        help='Agent type to use for benchmarking'
    )

    return parser