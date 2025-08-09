"""Command execution handlers for EmbodiedAgentSim CLI"""
import traceback

from easim.agents.sample import SampleAgent
from easim.benchmark.benchmark import HabitatBenchmark
from easim.utils.constants import BENCHMARK_CONFIG, AGENT_LIST

from easim.demo.interactive import interactive
from easim.utils.habitat_utils import setup_habitat_lab_env

# Set up habitat-lab environment
setup_habitat_lab_env()




def execute_command(args):
    """Execute the appropriate command based on arguments"""
    print("EmbodiedAgentSim - Habitat-based Simulation Framework")
    print(f"Command: {args.command}")
    print("-" * 50)

    try:
        if args.command == 'interactive':
            interactive()
        elif args.command == 'list-tasks':
            list_tasks()
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

def list_tasks():
    """List all available benchmark tasks"""
    print("Available Benchmark Tasks:")
    print("=" * 40)
    
    # Group tasks by category
    nav_tasks = {k: v for k, v in BENCHMARK_CONFIG.items() if 'nav/' in v}
    rearrange_tasks = {k: v for k, v in BENCHMARK_CONFIG.items() if 'rearrange/' in v}
    multi_agent_tasks = {k: v for k, v in BENCHMARK_CONFIG.items() if 'multi_agent/' in v}
    
    print("\nNavigation Tasks:")
    print("-" * 20)
    for task in sorted(nav_tasks.keys()):
        print(f"  • {task}")
    
    print("\nRearrange Tasks:")
    print("-" * 20)
    for task in sorted(rearrange_tasks.keys()):
        print(f"  • {task}")
    
    print("\nMulti-Agent Tasks:")
    print("-" * 20)
    for task in sorted(multi_agent_tasks.keys()):
        print(f"  • {task}")
    
    print(f"\nTotal: {len(BENCHMARK_CONFIG)} tasks available")

def benchmark(args):
    """Run benchmark evaluation"""
    print(f"Task: {args.task}")
    print(f"Episodes: {args.episodes if args.episodes is not None else 'all'}")
    print(f"Agent: {args.agent}")
    print(f"Record video: {args.record}")
    print("-" * 30)
    

    
    # Initialize benchmark
    try:
        habitat_benchmark = HabitatBenchmark(task_name=args.task, agent=args.agent)
    except KeyError:
        print(f"Error: Task '{args.task}' not found in benchmark configurations")
        return 1
    
    # Run evaluation
    print(f"Starting evaluation...")
    habitat_benchmark.evaluate(
        num_episodes=args.episodes, 
        enable_record=args.record
    )


    
    return 0