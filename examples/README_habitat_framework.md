# Habitat Lab Framework Guide with Video Recording & Interactive Play

This guide teaches you how to use Habitat Lab as a framework for running scene and task datasets with clean wrapper interfaces, plus video recording and interactive play capabilities.

## Quick Start

```python
from easim.core.habitat_wrapper import create_habitat_env

# Create environment
env = create_habitat_env(task="pointnav", dataset="mp3d")

# Use like any RL environment
obs = env.reset()
obs, reward, done, info = env.step('move_forward')
env.close()
```

## Key Components

### 1. Environment Wrapper (`HabitatEnvironmentWrapper`)

The main interface that wraps Habitat Lab environments:

```python
from easim.core.habitat_wrapper import HabitatEnvironmentWrapper, HabitatWrapperConfig

config = HabitatWrapperConfig(
    task_type=TaskType.POINTNAV,
    dataset_type=DatasetType.MP3D,
    width=256,
    height=256,
    forward_step_size=0.25,
    turn_angle=30
)

env = HabitatEnvironmentWrapper(config)
```

**Key Methods:**
- `reset()` - Start new episode
- `step(action)` - Take action, returns (obs, reward, done, info)
- `get_scene_info()` - Get current scene information
- `get_episode_stats()` - Get episode statistics
- `start_recording(path)` - Start video recording
- `stop_recording()` - Stop video recording
- `record_episode(strategy, path)` - Record episode with navigation strategy
- `run_interactive()` - Start interactive pygame control
- `close()` - Cleanup resources

### 2. Task Dataset Manager (`TaskDatasetManager`)

Manages available datasets and creates environments:

```python
from easim.core.habitat_wrapper import TaskDatasetManager

manager = TaskDatasetManager()

# Discover available datasets
datasets = manager.list_datasets()  # ['mp3d', 'hm3d']

# Get dataset information
info = manager.get_dataset_info('mp3d')

# Create environment
env = manager.create_environment('pointnav', 'mp3d')
```

### 3. Configuration System

Clean configuration with sensible defaults:

```python
@dataclass
class HabitatWrapperConfig:
    task_type: TaskType = TaskType.POINTNAV
    dataset_type: DatasetType = DatasetType.MP3D
    scene_dataset_path: Optional[str] = None
    
    # Simulator settings
    width: int = 256
    height: int = 256
    forward_step_size: float = 0.25
    turn_angle: int = 30
    
    # Episode settings
    max_episode_length: int = 500
```

## Supported Tasks

### PointNav (Point Navigation)
Navigate to GPS coordinates.

```python
env = create_habitat_env(task="pointnav", dataset="mp3d")
obs = env.reset()
# obs contains: 'rgb', 'depth', 'pointgoal' (distance, angle)
```

### ObjectNav (Object Navigation)  
Navigate to find specific objects.

```python
env = create_habitat_env(task="objectnav", dataset="mp3d")
obs = env.reset()
# obs contains: 'rgb', 'depth', 'objectgoal', 'gps', 'compass'
```

## Supported Datasets

### MP3D (Matterport3D)
- Real-world indoor scenes
- Tasks: PointNav, ObjectNav, ImageNav
- High-quality textures and geometry

### HM3D (Habitat-Matterport 3D)
- Largest dataset of indoor scenes
- Tasks: PointNav, ObjectNav
- Diverse scene types

## Action Space

Standard discrete actions:
- `0` or `'stop'` - Stop/finish episode
- `1` or `'move_forward'` - Move forward 0.25m (configurable)
- `2` or `'turn_left'` - Turn left 30° (configurable) 
- `3` or `'turn_right'` - Turn right 30° (configurable)
- `4` or `'look_up'` - Tilt camera up (ObjectNav only)
- `5` or `'look_down'` - Tilt camera down (ObjectNav only)

## Observation Space

Common observations across tasks:
- `'rgb'` - RGB camera image (H, W, 3)
- `'depth'` - Depth image (H, W, 1)

Task-specific observations:
- **PointNav:** `'pointgoal'` - [distance, angle] to goal
- **ObjectNav:** `'objectgoal'` - object category, `'gps'`, `'compass'`

## Example Usage Patterns

### 1. Random Agent
```python
import numpy as np

env = create_habitat_env("pointnav", "mp3d")

for episode in range(10):
    obs = env.reset()
    
    while True:
        action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
        obs, reward, done, info = env.step(action)
        
        if done:
            success = info.get('success', 0) > 0.5
            print(f"Episode {episode}: Success={success}")
            break

env.close()
```

### 2. Distance-based Agent
```python
def simple_pointnav_agent(obs):
    goal = obs['pointgoal']
    distance, angle = goal[0], goal[1]
    
    if distance < 0.2:
        return 'stop'
    elif abs(angle) > 0.3:
        return 'turn_left' if angle > 0 else 'turn_right'
    else:
        return 'move_forward'

env = create_habitat_env("pointnav", "mp3d")
obs = env.reset()

while True:
    action = simple_pointnav_agent(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

### 3. Multi-Episode Training
```python
def train_agent(num_episodes=100):
    env = create_habitat_env("pointnav", "mp3d", seed=42)
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        while True:
            # Your agent logic here
            action = your_agent.act(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Train your agent here
            your_agent.learn(obs, action, reward, done)
            
            if done:
                break
        
        if episode % 10 == 0:
            success_rate = compute_success_rate()
            print(f"Episode {episode}: Success rate = {success_rate:.2f}")
    
    env.close()
```

## Video Recording Features

### 1. Manual Recording
```python
env = create_habitat_env("pointnav", "mp3d")

# Start recording
env.start_recording("output/my_video.mp4", fps=30)

obs = env.reset()
for _ in range(50):
    obs, reward, done, info = env.step('move_forward')
    if done:
        break

# Stop recording
video_path = env.stop_recording()
print(f"Video saved to: {video_path}")
```

### 2. Strategy-based Recording
```python
from easim.core.video_recorder import BaseNavigationStrategy

class MyStrategy(BaseNavigationStrategy):
    def get_next_action(self, obs, step):
        # Your navigation logic here
        return 'move_forward'
    
    def is_done(self, obs, step):
        return step >= 100

# Record episode with strategy
strategy = MyStrategy()
result = env.record_episode(
    strategy=strategy,
    output_path="output/strategy_video.mp4",
    fps=30,
    save_frames=True  # Also save individual frames
)

print(f"Success: {result['episode_stats']['success']}")
print(f"Video: {result['video_path']}")
```

### 3. Built-in Navigation Strategies
```python
from easim.core.habitat_wrapper import (
    HabitatRandomStrategy,
    HabitatPointNavStrategy,
    HabitatExplorationStrategy
)

# Random navigation
random_strategy = HabitatRandomStrategy(max_steps=100, forward_prob=0.7)

# Point navigation (for PointNav task)
pointnav_strategy = HabitatPointNavStrategy(max_steps=200, success_distance=0.2)

# Exploration with obstacle avoidance
exploration_strategy = HabitatExplorationStrategy(max_steps=150)

# Record with any strategy
result = env.record_episode(pointnav_strategy, "pointnav_demo.mp4")
```

## Interactive Play

### 1. Basic Interactive Mode
```python
env = create_habitat_env("pointnav", "mp3d")

# Start interactive mode with pygame
env.run_interactive(window_size=(800, 600))

# Controls:
# W/↑ - Move Forward
# A/← - Turn Left  
# D/→ - Turn Right
# ESC - Quit
```

### 2. Interactive with Recording
```python
env = create_habitat_env("pointnav", "mp3d")

# Start recording
env.start_recording("interactive_session.mp4")

# Play interactively (recording in background)
env.run_interactive()

# Stop recording when done
video_path = env.stop_recording()
```

## Advanced Features

### Custom Scene Dataset
```python
config = HabitatWrapperConfig(
    scene_dataset_path="/path/to/your/dataset.scene_dataset_config.json"
)
env = HabitatEnvironmentWrapper(config)
```

### High-Resolution Rendering
```python
config = HabitatWrapperConfig(
    width=512,
    height=512
)
env = HabitatEnvironmentWrapper(config)
```

### Custom Movement Parameters
```python
config = HabitatWrapperConfig(
    forward_step_size=0.5,  # Larger steps
    turn_angle=45           # Sharper turns
)
env = HabitatEnvironmentWrapper(config)
```

## Integration with Your Project

The wrapper integrates seamlessly with your existing simulator:

```python
# In your existing code
from easim.core.habitat_wrapper import create_habitat_env

# Replace your simulator creation
# old: simulator = CoreSimulator(config)
env = create_habitat_env(task="pointnav", dataset="mp3d")

# Use same interface
obs = env.reset()
obs, reward, done, info = env.step(action)
```

## File Structure

```
easim/
├── core/
│   ├── habitat_wrapper.py     # Main wrapper classes
│   └── simulator.py           # Your existing simulator
├── utils/
│   └── constants.py           # Dataset paths and constants
└── examples/
    ├── habitat_tutorial.py    # Complete tutorial script
    └── README_habitat_framework.md  # This guide
```

## Combined Usage Examples

### 1. Training with Video Logging
```python
def train_with_video_logging(num_episodes=100):
    env = create_habitat_env("pointnav", "mp3d", seed=42)
    
    for episode in range(num_episodes):
        # Record every 10th episode
        if episode % 10 == 0:
            video_path = f"training/episode_{episode}.mp4"
            env.start_recording(video_path)
        
        obs = env.reset()
        while True:
            action = your_agent.act(obs)
            obs, reward, done, info = env.step(action)
            your_agent.learn(obs, action, reward, done)
            
            if done:
                break
        
        # Stop recording if active
        if env.is_recording():
            env.stop_recording()
    
    env.close()
```

### 2. Multi-Strategy Comparison
```python
def compare_strategies():
    env = create_habitat_env("pointnav", "mp3d")
    
    strategies = [
        ("Random", HabitatRandomStrategy(max_steps=100)),
        ("PointNav", HabitatPointNavStrategy(max_steps=200)),
        ("Exploration", HabitatExplorationStrategy(max_steps=150))
    ]
    
    results = []
    for name, strategy in strategies:
        video_path = f"comparison/{name.lower()}_strategy.mp4"
        result = env.record_episode(strategy, video_path)
        
        results.append({
            'strategy': name,
            'success': result['episode_stats']['success'],
            'steps': result['total_steps'],
            'video': result['video_path']
        })
    
    return results
```

### 3. Interactive Debugging
```python
def debug_agent_behavior():
    env = create_habitat_env("pointnav", "mp3d")
    
    # First, record your agent's behavior
    agent_strategy = YourAgentStrategy()
    env.record_episode(agent_strategy, "debug/agent_behavior.mp4")
    
    # Then play interactively to compare
    print("Now play manually to see how you would solve it...")
    env.start_recording("debug/manual_play.mp4")
    env.run_interactive()
    env.stop_recording()
    
    env.close()
```

## Installation & Dependencies

```bash
# Core dependencies
pip install habitat-sim habitat-lab

# For video recording
pip install opencv-python

# For interactive play
pip install pygame

# For visualization
pip install matplotlib

# Optional: for better video codecs
pip install ffmpeg-python
```

## Next Steps

1. **Run the tutorial**: `python examples/habitat_tutorial.py`
2. **Try interactive mode**: Experience manual control with real-time rendering
3. **Record training videos**: Use video recording to debug and showcase your agents
4. **Integrate with your agents**: Use the wrapper in your existing agent classes
5. **Customize for your needs**: Modify `HabitatWrapperConfig` for your requirements
6. **Create navigation strategies**: Implement custom `BaseNavigationStrategy` classes
7. **Scale up**: Use the framework for large-scale training with automated recording

## Troubleshooting

**Environment won't start:**
- Check dataset paths in `easim/utils/constants.py`
- Ensure scene dataset files exist
- Verify Habitat Lab installation

**Performance issues:**
- Reduce image resolution in config
- Disable physics if not needed
- Use appropriate GPU settings

**Import errors:**
- Ensure Habitat Lab is properly installed
- Check Python path includes your project directory