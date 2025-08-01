"""
Example script for running object navigation with HM3D scenes.
"""

from easim.core.object_nav_pipeline import ObjectNavPipeline


def run_interactive_episode():
    """Run a single interactive episode with visual display."""
    # Use pointnav test config since objectnav HM3D data is not available
    with ObjectNavPipeline("benchmark/nav/pointnav/pointnav_habitat_test.yaml") as pipeline:
        pipeline.set_agent("heuristic")
        print("Starting interactive object navigation episode...")
        print("Press 'q' to quit, 's' to stop episode")
        
        stats = pipeline.run_episode(display=True, max_steps=500)
        print(f"Episode completed with success: {stats['success']}")


def run_batch_evaluation():
    """Run multiple episodes for evaluation."""
    # Use pointnav test config since objectnav HM3D data is not available
    with ObjectNavPipeline("benchmark/nav/pointnav/pointnav_habitat_test.yaml") as pipeline:
        print("Running batch evaluation with random agent...")
        random_results = pipeline.run_multiple_episodes(
            num_episodes=5, 
            agent_type="random", 
            display=False
        )
        
        print("\nRunning batch evaluation with heuristic agent...")
        heuristic_results = pipeline.run_multiple_episodes(
            num_episodes=5,
            agent_type="heuristic", 
            display=False
        )
        
        # Compare agents
        random_success = sum(1 for ep in random_results if ep['success']) / len(random_results)
        heuristic_success = sum(1 for ep in heuristic_results if ep['success']) / len(heuristic_results)
        
        print(f"\n=== Agent Comparison ===")
        print(f"Random Agent Success Rate: {random_success:.2%}")
        print(f"Heuristic Agent Success Rate: {heuristic_success:.2%}")


def main():
    """Main function to demonstrate object navigation."""
    print("Object Navigation Demo")
    print("1. Interactive episode")
    print("2. Batch evaluation")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        run_interactive_episode()
    elif choice == "2":
        run_batch_evaluation()
    else:
        print("Invalid choice. Running interactive episode...")
        run_interactive_episode()


if __name__ == "__main__":
    main()