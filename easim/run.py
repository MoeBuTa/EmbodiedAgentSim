"""
Minimal entry point for EmbodiedAgentSim CLI
"""
from easim.cli.parser import create_parser
from easim.cli.commands import execute_command


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 1

    return execute_command(args)


if __name__ == "__main__":
    exit(main())