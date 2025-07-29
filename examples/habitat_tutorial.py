"""
Tutorial: Using Habitat Lab with Clean Wrapper Interface

This script demonstrates how to use the Habitat Lab framework for running
scene and task datasets with clean wrapper interfaces.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path

# Import our wrapper and navigation strategies
from easim.core.habitat_wrapper import (
    create_habitat_env, 
    HabitatWrapperConfig, 
    TaskType, 
    DatasetType,
    TaskDatasetManager,
    HabitatRandomStrategy,
    HabitatPointNavStrategy,
    HabitatExplorationStrategy
)
from easim.utils.constants import OUTPUT_DIR


def basic_environment_usage():
    """Demonstrate basic environment usage"""
    print("=== Basic Environment Usage ===")
    
    # Create environment with simple interface
    env = create_habitat_env(task="pointnav", dataset="mp3d")
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"RGB shape: {obs['rgb'].shape}")
        
        # Take some actions
        actions = ['move_forward', 'turn_left', 'move_forward', 'turn_right']
        
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}")
            
            if done:
                print("Episode finished!")
                break
        
        # Print episode statistics
        stats = env.get_episode_stats()
        print(f"Episode stats: {stats}")
        
    finally:
        env.close()


def advanced_configuration():
    """Demonstrate advanced configuration options"""
    print("\n=== Advanced Configuration ===")
    
    # Create custom configuration
    config = HabitatWrapperConfig(
        task_type=TaskType.OBJECTNAV,
        dataset_type=DatasetType.MP3D,
        width=512,
        height=512,
        forward_step_size=0.5,  # Larger steps
        turn_angle=45,          # Larger turns
        max_episode_length=200,
        seed=42
    )
    
    from easim.core.habitat_wrapper import HabitatEnvironmentWrapper
    env = HabitatEnvironmentWrapper(config)
    
    try:
        obs = env.reset()
        scene_info = env.get_scene_info()
        print(f"Scene info: {scene_info}")
        
        # Test different action types
        for action_type in [0, 1, 2, 3]:  # stop, forward, left, right
            obs, reward, done, info = env.step(action_type)
            print(f"Action {action_type}: Success metric = {info.get('success', 0)}")
            
            if done:
                break
                
    finally:
        env.close()


def dataset_discovery():
    """Demonstrate dataset discovery and management"""
    print("\n=== Dataset Discovery ===")
    
    manager = TaskDatasetManager()
    
    # List available datasets
    datasets = manager.list_datasets()
    print(f"Available datasets: {datasets}")
    
    # Get dataset info
    for dataset_name in datasets:
        info = manager.get_dataset_info(dataset_name)
        print(f"\n{dataset_name.upper()}:")
        print(f"  Path: {info['path']}")
        print(f"  Tasks: {info['tasks']}")
        print(f"  Scenes: {info['scenes']}")


def multi_episode_training_loop():
    """Demonstrate training loop with multiple episodes"""
    print("\n=== Multi-Episode Training Loop ===")
    
    env = create_habitat_env(task="pointnav", dataset="mp3d", seed=42)
    
    try:
        num_episodes = 3
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            
            print(f"\nEpisode {episode + 1}")
            
            while True:
                # Simple random policy
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                if step_count % 20 == 0:
                    print(f"  Step {step_count}: Distance to goal = {info.get('distance_to_goal', 'N/A')}")
                
                if done:
                    success = info.get('success', 0) > 0.5
                    print(f"  Episode finished: Success={success}, Steps={step_count}, Reward={episode_reward:.3f}")
                    break
                
                if step_count >= 100:  # Prevent infinite loops
                    print("  Episode truncated at 100 steps")
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"  Average reward: {np.mean(episode_rewards):.3f}")
        print(f"  Average length: {np.mean(episode_lengths):.1f}")
        
    finally:
        env.close()


def visualize_observations(save_images: bool = False):
    """Demonstrate observation visualization"""
    print("\n=== Observation Visualization ===")
    
    env = create_habitat_env(
        task="objectnav", 
        dataset="mp3d",
        width=256,
        height=256
    )
    
    try:
        obs = env.reset()
        
        # Take a few steps to get interesting observations
        for _ in range(5):
            obs, _, done, _ = env.step('move_forward')
            if done:
                obs = env.reset()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB observation
        if 'rgb' in obs:
            axes[0].imshow(obs['rgb'])
            axes[0].set_title('RGB Observation')
            axes[0].axis('off')
        
        # Depth observation
        if 'depth' in obs:
            depth_img = obs['depth'].squeeze()
            axes[1].imshow(depth_img, cmap='viridis')
            axes[1].set_title('Depth Observation')
            axes[1].axis('off')
        
        # Goal information (if available)
        if 'objectgoal' in obs:
            # This would be object category information
            axes[2].text(0.1, 0.5, f"Object Goal:\n{obs['objectgoal']}", 
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Goal Information')
            axes[2].axis('off')
        elif 'pointgoal' in obs:
            # Point goal information
            goal = obs['pointgoal']
            axes[2].text(0.1, 0.5, f"Point Goal:\nDistance: {goal[0]:.2f}\nAngle: {goal[1]:.2f}", 
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Goal Information')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_images:
            output_dir = Path("output/habitat_tutorial")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "observations_demo.png", dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to {output_dir / 'observations_demo.png'}")
        
        plt.show()
        
    finally:
        env.close()


def video_recording_demo():
    """Demonstrate video recording capabilities"""
    print("\n=== Video Recording Demo ===")
    
    env = create_habitat_env(task="pointnav", dataset="mp3d", seed=42)
    
    try:
        # Create output directory
        video_dir = OUTPUT_DIR / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Demo 1: Manual recording
        print("\nDemo 1: Manual start/stop recording")
        video_path1 = video_dir / "manual_recording.mp4"
        
        env.start_recording(str(video_path1), fps=30)
        obs = env.reset()
        
        # Take some actions while recording
        actions = ['move_forward'] * 5 + ['turn_right'] * 2 + ['move_forward'] * 3
        for action in actions:
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        saved_path = env.stop_recording()
        print(f"  Video saved to: {saved_path}")
        
        # Demo 2: Strategy-based recording
        print("\nDemo 2: Strategy-based recording")
        video_path2 = video_dir / "random_strategy.mp4"
        
        strategy = HabitatRandomStrategy(max_steps=50, forward_prob=0.7)
        result = env.record_episode(
            strategy=strategy,
            output_path=str(video_path2),
            fps=30,
            save_frames=True
        )
        
        print(f"  Recording completed:")
        print(f"    Video: {result['video_path']}")
        print(f"    Duration: {result['duration']:.2f}s")
        print(f"    Steps: {result['total_steps']}")
        print(f"    Success: {result['episode_stats']['success']}")
        
        # Demo 3: Point navigation strategy recording
        print("\nDemo 3: Point navigation strategy recording")
        video_path3 = video_dir / "pointnav_strategy.mp4"
        
        strategy = HabitatPointNavStrategy(max_steps=100)
        result = env.record_episode(
            strategy=strategy,
            output_path=str(video_path3),
            fps=30
        )
        
        print(f"  PointNav recording completed:")
        print(f"    Video: {result['video_path']}")
        print(f"    Success: {result['episode_stats']['success']}")
        print(f"    Final distance: {result['episode_stats'].get('final_distance', 'N/A')}")
        
    finally:
        env.close()


def interactive_play_demo():
    """Demonstrate interactive play functionality"""
    print("\n=== Interactive Play Demo ===")
    print("This will open an interactive window where you can control the agent.")
    print("Controls:")
    print("  W/↑ - Move Forward")
    print("  A/← - Turn Left")
    print("  D/→ - Turn Right")
    print("  ESC - Quit")
    print("\nPress Enter to start interactive mode, or 'skip' to skip...")
    
    user_input = input().strip().lower()
    if user_input == 'skip':
        print("Skipping interactive demo")
        return
    
    env = create_habitat_env(task="pointnav", dataset="mp3d")
    
    try:
        # Optional: start recording during interactive play
        video_dir = OUTPUT_DIR / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        print("Starting recording during interactive play...")
        env.start_recording(str(video_dir / "interactive_play.mp4"))
        
        # Start interactive mode
        env.run_interactive(window_size=(800, 600))
        
        # Stop recording
        video_path = env.stop_recording()
        print(f"Interactive session recorded to: {video_path}")
        
    except Exception as e:
        print(f"Interactive mode error: {e}")
        print("Make sure pygame is installed: pip install pygame")
    finally:
        env.close()


def advanced_strategies_demo():
    """Demonstrate advanced navigation strategies"""
    print("\n=== Advanced Navigation Strategies ===")
    
    env = create_habitat_env(task="pointnav", dataset="mp3d", seed=42)
    video_dir = OUTPUT_DIR / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        strategies = [
            ("Random", HabitatRandomStrategy(max_steps=50)),
            ("PointNav", HabitatPointNavStrategy(max_steps=100)),
            ("Exploration", HabitatExplorationStrategy(max_steps=75))
        ]
        
        results = []
        
        for name, strategy in strategies:
            print(f"\nTesting {name} strategy:")
            
            video_path = video_dir / f"{name.lower()}_strategy.mp4"
            result = env.record_episode(
                strategy=strategy,
                output_path=str(video_path),
                fps=30
            )
            
            success = result['episode_stats']['success']
            steps = result['total_steps']
            duration = result['duration']
            
            print(f"  Success: {success}")
            print(f"  Steps: {steps}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Video: {result['video_path']}")
            
            results.append((name, success, steps, duration))
        
        # Summary comparison
        print("\nStrategy Comparison:")
        print(f"{'Strategy':<12} {'Success':<8} {'Steps':<6} {'Duration':<8}")
        print("-" * 40)
        for name, success, steps, duration in results:
            print(f"{name:<12} {success:<8} {steps:<6} {duration:<8.2f}")
            
    finally:
        env.close()


def multi_task_recording():
    """Record videos for multiple tasks"""
    print("\n=== Multi-Task Recording ===")
    
    tasks = ["pointnav", "objectnav"]
    video_dir = OUTPUT_DIR / "videos" / "multi_task"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    for task in tasks:
        print(f"\nRecording {task.upper()} task:")
        
        try:
            env = create_habitat_env(task=task, dataset="mp3d", seed=42)
            
            # Choose appropriate strategy
            if task == "pointnav":
                strategy = HabitatPointNavStrategy(max_steps=100)
            else:
                strategy = HabitatExplorationStrategy(max_steps=100)
            
            video_path = video_dir / f"{task}_demo.mp4"
            result = env.record_episode(
                strategy=strategy,
                output_path=str(video_path),
                fps=30
            )
            
            print(f"  Video saved: {result['video_path']}")
            print(f"  Success: {result['episode_stats']['success']}")
            print(f"  Steps: {result['total_steps']}")
            
            env.close()
            
        except Exception as e:
            print(f"  Error recording {task}: {e}")


def compare_tasks():
    """Compare different task types"""
    print("\n=== Task Comparison ===")
    
    tasks = ["pointnav", "objectnav"]
    
    for task in tasks:
        print(f"\n{task.upper()}:")
        
        try:
            env = create_habitat_env(task=task, dataset="mp3d")
            
            obs = env.reset()
            scene_info = env.get_scene_info()
            
            print(f"  Observation keys: {list(obs.keys())}")
            print(f"  Action space: {env.action_space}")
            print(f"  Scene ID: {scene_info.get('scene_id', 'unknown')}")
            
            # Take one step to see metrics
            obs, reward, done, info = env.step('move_forward')
            print(f"  Available metrics: {list(info.keys())}")
            
            env.close()
            
        except Exception as e:
            print(f"  Error creating {task} environment: {e}")


def main():
    """Run all tutorial examples"""
    print("Habitat Lab Framework Tutorial with Video Recording & Interactive Play")
    print("=" * 70)
    
    # Run all examples
    basic_environment_usage()
    advanced_configuration()
    dataset_discovery()
    multi_episode_training_loop()
    compare_tasks()
    
    # New features
    video_recording_demo()
    advanced_strategies_demo()
    multi_task_recording()
    
    # Interactive demo (optional)
    interactive_play_demo()
    
    # Optional visualization (requires matplotlib)
    try:
        visualize_observations(save_images=True)
    except ImportError:
        print("\nSkipping visualization (matplotlib not available)")
    except Exception as e:
        print(f"\nSkipping visualization due to error: {e}")
    
    print("\n" + "=" * 70)
    print("Tutorial completed!")
    print("\nFeatures demonstrated:")
    print("✓ Basic Habitat Lab environment usage")
    print("✓ Advanced configuration options")
    print("✓ Dataset discovery and management")
    print("✓ Multi-episode training loops")
    print("✓ Video recording with multiple strategies")
    print("✓ Interactive play with pygame")
    print("✓ Advanced navigation strategies")
    print("✓ Multi-task recording")
    print("\nNext steps:")
    print("1. Modify HabitatWrapperConfig for your specific needs")
    print("2. Implement your own navigation strategies")
    print("3. Create custom agents using the clean interface")
    print("4. Use video recording for debugging and demonstrations")
    print("5. Try interactive mode for manual exploration")
    print("6. Scale up for large-scale training with automated recording")


if __name__ == "__main__":
    main()