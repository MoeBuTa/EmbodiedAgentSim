#!/usr/bin/env python3
"""
R2R Dataset Usage Examples
"""

from easim.datasets.r2r_loader import R2RDatasetLoader, download_r2r_scenes
from easim.sim.r2r_environment import R2REnvironment


def example_1_download_and_explore():
    """Example 1: Download R2R dataset and explore it"""
    print("=== Example 1: Download and Explore R2R ===")

    # Create loader
    loader = R2RDatasetLoader()

    # Download dataset (automatic)
    print("Downloading R2R dataset...")
    loader.download_r2r_data()

    # Show dataset statistics
    loader.print_dataset_stats()

    # Load a specific split
    val_episodes = loader.load_split("val_seen")
    print(f"\nLoaded {len(val_episodes)} validation episodes")

    # Show first episode
    first_episode = val_episodes[0]
    print(f"\nFirst episode: {first_episode.episode_id}")
    print(f"Scene: {first_episode.scan}")
    print(f"Instructions:")
    for i, instruction in enumerate(first_episode.instructions):
        print(f"  {i + 1}. {instruction}")
    print(f"Path length: {first_episode.distance:.1f}m")
    print(f"Path points: {len(first_episode.path)}")


def example_2_scene_requirements():
    """Example 2: Check scene requirements"""
    print("\n=== Example 2: R2R Scene Requirements ===")

    # Get required scenes
    required_scenes = download_r2r_scenes()

    print(f"\nR2R requires {len(required_scenes)} MP3D scenes")
    print("First 10 scenes:", required_scenes[:10])


def example_3_basic_navigation():
    """Example 3: Basic R2R navigation"""
    print("\n=== Example 3: Basic R2R Navigation ===")

    try:
        # Create environment
        env = R2REnvironment(split="val_seen")

        # Reset to first episode
        state = env.reset(episode_idx=0)
        env.print_episode_info()

        # Check if we have a simulator (scene available)
        if env.simulator is None:
            print("⚠️ No MP3D scenes available. Download scenes first:")
            print("python -m habitat_sim.utils.datasets_download --uids mp3d --data-path data")
            return

        # Take some navigation actions
        actions = ["move_forward", "turn_left", "move_forward", "turn_right", "move_forward"]

        for i, action in enumerate(actions):
            print(f"\nStep {i + 1}: {action}")
            state, reward, done, info = env.step(action)

            print(f"  Reward: {reward:.2f}")
            print(f"  Distance to target: {info['distance_to_target']:.1f}m")
            print(f"  Success: {info['success']}")

            if done:
                print("  Episode completed!")
                break

        # Try next instruction variant
        print(f"\nSwitching to next instruction...")
        env.next_instruction()
        print(f"New instruction: {env.get_current_instruction()}")

        # Try next episode
        print(f"\nMoving to next episode...")
        env.next_episode()
        env.print_episode_info()

        # Cleanup
        if env.simulator:
            env.simulator.close()

    except Exception as e:
        print(f"Error in navigation example: {e}")


def example_4_instruction_analysis():
    """Example 4: Analyze R2R instructions"""
    print("\n=== Example 4: R2R Instruction Analysis ===")

    loader = R2RDatasetLoader()

    # Load episodes
    episodes = loader.load_split("val_seen")

    # Analyze instructions
    all_instructions = []
    for episode in episodes[:10]:  # First 10 episodes
        all_instructions.extend(episode.instructions)

    print(f"Analyzing {len(all_instructions)} instructions from 10 episodes...")

    # Find common words
    import re
    words = []
    for instruction in all_instructions:
        words.extend(re.findall(r'\b\w+\b', instruction.lower()))

    from collections import Counter
    common_words = Counter(words).most_common(10)

    print("\nMost common words in instructions:")
    for word, count in common_words:
        print(f"  {word}: {count}")

    print("\nSample instructions:")
    for i, instruction in enumerate(all_instructions[:5]):
        print(f"  {i + 1}. {instruction}")


def example_5_command_line_usage():
    """Example 5: Show command line usage"""
    print("\n=== Example 5: Command Line Usage ===")

    print("You can also use R2R from command line:")
    print()
    print("# Download R2R dataset")
    print("python -m easim.datasets.r2r_loader")
    print()
    print("# Run R2R environment demo")
    print("python -m easim.sim.r2r_environment")
    print()
    print("# Use with easim command (if we add R2R support)")
    print("# easim r2r --split val_seen --episodes 5")


def main():
    """Run all examples"""
    print("R2R Dataset Loading and Usage Examples")
    print("=" * 50)

    # Run examples
    example_1_download_and_explore()
    example_2_scene_requirements()
    example_3_basic_navigation()
    example_4_instruction_analysis()
    example_5_command_line_usage()

    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("\nNext steps:")
    print("1. Download MP3D scenes for full R2R functionality")
    print("2. Implement your own navigation agent")
    print("3. Evaluate on R2R metrics (SPL, success rate, etc.)")


if __name__ == "__main__":
    main()