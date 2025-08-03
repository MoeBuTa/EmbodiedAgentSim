"""
Command line argument parser for EmbodiedAgentSim
"""
import argparse
from easim.utils.constants import OUTPUT_DIR


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="EmbodiedAgentSim - Habitat-based simulation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test basic habitat-lab setup
  easim test --dataset MP3D

  # Record random navigation video
  easim record --dataset HM3D --output-dir videos
        """
    )

    # Add subcommands instead of mode
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test basic habitat-lab setup')
    test_parser.add_argument('--dataset', choices=['MP3D', 'HM3D'], default='MP3D')
    test_parser.add_argument('--scene-path', help='Custom scene path')

    # Record command
    record_parser = subparsers.add_parser('record', help='Record navigation video')
    record_parser.add_argument('--dataset', choices=['MP3D', 'HM3D'], default='MP3D')
    record_parser.add_argument('--scene-path', help='Custom scene path')
    record_parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Output directory')
    record_parser.add_argument('--video-name', default='simulation.mp4', help='Video filename')
    record_parser.add_argument('--max-steps', type=int, default=100, help='Max navigation steps')
    record_parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    record_parser.add_argument('--save-frames', action='store_true', help='Save individual frames')

    return parser