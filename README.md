# EmbodiedAgentSim

A clean, object-oriented framework for embodied AI simulation using Habitat-Lab. Features video recording, interactive control, and support for multiple datasets.

## Features

- 🤖 **Multi-Dataset Support**: MP3D and HM3D scenes
- 🎥 **Video Recording**: Automatic MP4 generation from navigation
- 🎮 **Interactive Control**: Real-time pygame-based control
- 🏗️ **Clean OOP Design**: Extensible simulator and strategy classes
- 📁 **Portable Setup**: No hardcoded paths, works anywhere
- 🔧 **Graceful Degradation**: Works even with missing optional dependencies

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd EmbodiedAgentSim

# Create conda environment
conda env create -f environment.yml
conda activate habitat

# Install package
pip install -e .
```

### Download Test Data

```bash
# Download MP3D test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data

# Optional: Download example objects
python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path data
```

### Basic Usage

```bash
# Test basic simulator
easim simulator --dataset MP3D

# Record navigation video
easim record --dataset MP3D --max-steps 100 --video-name exploration.mp4

# Interactive control (requires pygame)
easim interactive --dataset MP3D
```

## Commands

### `easim simulator`
Test basic simulator functionality with different datasets.

```bash
easim simulator --dataset MP3D              # Test MP3D simulator
easim simulator --dataset HM3D              # Test HM3D simulator  
easim simulator --scene-path path/to/scene  # Custom scene
```

### `easim record`
Record navigation videos with different strategies.

```bash
# Basic recording
easim record --dataset MP3D --max-steps 200

# Custom output
easim record --dataset HM3D --output-dir videos --video-name my_nav.mp4

# Longer exploration
easim record --dataset MP3D --max-steps 500 --video-name long_exploration.mp4
```

### `easim interactive`
Real-time interactive control with keyboard.

```bash
easim interactive --dataset MP3D

# Controls:
# W/↑ - Move Forward
# A/← - Turn Left  
# D/→ - Turn Right
# ESC - Quit
```

## Project Structure

```
EmbodiedAgentSim/
├── easim/
│   ├── sim/
│   │   ├── simulator.py          # Core simulator classes
│   │   ├── video_recorder.py     # Recording and strategies
│   │   └── task_environments.py  # Task environments (optional)
│   ├── utils/
│   │   ├── constants.py          # Path configurations
│   │   └── config.py             # Habitat configurations
│   ├── examples/
│   │   └── use_cases.py          # Demo functions
│   └── run.py                    # Main entry point
├── data/
│   ├── scene_datasets/           # Scene files
│   └── output/                   # Generated videos
├── environment.yml               # Conda environment
└── setup.py                     # Package setup
```

## Core Components

### Simulators

```python
from easim.sim.simulator import SimulatorFactory

# Create simulators
mp3d_sim = SimulatorFactory.create_simulator("MP3D")
hm3d_sim = SimulatorFactory.create_simulator("HM3D")

# Custom simulator
from easim.sim.simulator import SimulatorConfig
config = SimulatorConfig(scene_path="path/to/scene.glb", scene_dataset="config.json")
custom_sim = SimulatorFactory.create_custom_simulator(config)
```

### Video Recording

```python
from easim.sim.video_recorder import SimulationRecorder, RandomNavigationStrategy

# Setup recording
recorder = SimulationRecorder(simulator, "output_dir")
strategy = RandomNavigationStrategy(max_steps=100, forward_prob=0.7)

# Record navigation
result = recorder.record_navigation(strategy, "video.mp4", save_frames=True)
```

### Navigation Strategies

```python
from easim.sim.video_recorder import FixedPathStrategy
from easim.utils.constants import ACTION_MOVE_FORWARD, ACTION_TURN_LEFT

# Fixed action sequence
actions = [ACTION_MOVE_FORWARD] * 10 + [ACTION_TURN_LEFT] * 3
strategy = FixedPathStrategy(actions)

# Custom strategy
class MyStrategy(BaseNavigationStrategy):
    def get_next_action(self, observations, step_count):
        # Your navigation logic
        return action
    
    def is_done(self, observations, step_count):
        # Termination condition  
        return done
```

## Advanced Usage

### Running Demos

```python
from easim.examples.use_cases import random_exploration_demo, interactive_pygame_demo

# Run specific demos
random_exploration_demo("MP3D", "output")
interactive_pygame_demo("HM3D")
```

### Custom Scenes

```bash
# Use custom scene file
easim simulator --scene-path data/my_scenes/custom.glb

# Record with custom scene
easim record --scene-path data/my_scenes/room.glb --video-name room_nav.mp4
```

## Dependencies

### Required
- `numpy` - Numerical computations
- `opencv-python` - Video recording
- `habitat-sim` - 3D simulation engine

### Optional
- `pygame` - Interactive control (install: `pip install pygame`)
- `habitat-lab` - Task environments (install: `pip install habitat-lab`)

### Installation Options

```bash
# Basic installation
pip install -e .

# With interactive support
pip install -e .[interactive]

# With habitat-lab tasks
pip install -e .[habitat]

# Full installation
pip install -e .[full]
```

## Configuration

The framework automatically detects your project structure. Key paths:

- **Project Root**: Auto-detected from package location
- **Data Directory**: `{project_root}/data/`
- **Output Directory**: `{project_root}/data/output/`
- **Scene Datasets**: `{project_root}/data/scene_datasets/`

No hardcoded paths - everything works relative to your project directory.

## Troubleshooting

### Scene Not Found
```bash
# Check if scene files exist
ls data/scene_datasets/mp3d_example/17DRP5sb8fy/

# Download test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data
```

### Import Errors
```bash
# Reinstall package
pip install -e .

# Check installation
easim --help
```

### Interactive Mode Issues
```bash
# Install pygame
pip install pygame

# For conda environments
conda install pygame
```

### Video Recording Issues
```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Reinstall if needed
pip install opencv-python
```

## Examples

### Basic Navigation Recording

```bash
# Quick test
easim record --dataset MP3D --max-steps 50

# Longer exploration  
easim record --dataset MP3D --max-steps 200 --video-name long_nav.mp4

# Custom output directory
easim record --dataset HM3D --output-dir my_videos --max-steps 150
```

### Interactive Exploration

```bash
# Launch interactive mode
easim interactive --dataset MP3D

# Use WASD or arrow keys to navigate
# Press ESC to quit
```

### Programmatic Usage

```python
from easim.sim.simulator import SimulatorFactory
from easim.sim.video_recorder import SimulationRecorder, RandomNavigationStrategy

# Create and test simulator
simulator = SimulatorFactory.create_simulator("MP3D")
observations = simulator.get_observations()
print(f"RGB shape: {observations['color_sensor'].shape}")

# Record navigation
recorder = SimulationRecorder(simulator, "output")
strategy = RandomNavigationStrategy(max_steps=100)
result = recorder.record_navigation(strategy, "test.mp4")
print(f"Recorded {result['total_steps']} steps")

simulator.close()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.