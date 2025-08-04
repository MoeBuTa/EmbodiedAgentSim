# EmbodiedAgentSim

A comprehensive framework for benchmarking embodied AI agents using Habitat-Lab. Supports 58+ benchmark tasks including Object Navigation, Point Navigation, Image Navigation, Rearrangement, and Multi-Agent scenarios with integrated video recording capabilities.

## 🎯 Features

- **58+ Benchmark Tasks**: Navigation, rearrangement, multi-agent scenarios
- **Video Recording**: Automatic episode recording with configurable output
- **Interactive Mode**: Real-time navigation testing and demonstration  
- **Modular Architecture**: Easy agent integration and task customization
- **Habitat-Lab Integration**: Full compatibility with Habitat ecosystem

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/EmbodiedAgentSim.git
cd EmbodiedAgentSim

# Create environment  
conda env create -f environment.yml
conda activate habitat

# Install package
pip install -e .
```

## 🚀 Quick Start

### List Available Tasks
```bash
# See all 58+ available benchmark tasks
easim list-tasks
```

### Run Benchmarks
```bash
# Basic benchmark evaluation
easim benchmark --task objectnav_hm3d --episodes 10

# With video recording
easim benchmark --task pointnav_mp3d --episodes 5 --video

# Different navigation tasks
easim benchmark --task imagenav_gibson --episodes 3
easim benchmark --task instance_imagenav_hm3d_v2 --episodes 5

# Rearrangement tasks
easim benchmark --task rearrange_multi_task --episodes 3
easim benchmark --task skill_pick --episodes 10

# Multi-agent scenarios
easim benchmark --task multi_agent_tidy_house --episodes 2 --video
```

### Interactive Demo
```bash
# Run interactive navigation demo
easim interactive
```

## 📋 Available Benchmark Categories

### Navigation Tasks
- **Object Navigation**: `objectnav_hm3d`, `objectnav_mp3d`, `objectnav_hssd_hab`
- **Point Navigation**: `pointnav_hm3d`, `pointnav_mp3d`, `pointnav_gibson`
- **Image Navigation**: `imagenav_mp3d`, `imagenav_gibson`
- **Instance Image Navigation**: `instance_imagenav_hm3d_v1`, `instance_imagenav_hm3d_v2`
- **VLN**: `vln_r2r`, `eqa_mp3d`

### Rearrangement Tasks
- **Skills**: `skill_pick`, `skill_place`, `skill_nav_to_obj`, `skill_open_fridge`
- **Multi-Task**: `rearrange_multi_task`, `set_table`, `prepare_groceries`, `tidy_house`
- **HAB3 Benchmarks**: `hab3_bench_single_agent`, `hab3_bench_multi_agent`
- **Play Scenarios**: `play_human`, `play_spot`, `play_stretch`

### Multi-Agent Tasks
- **Social Navigation**: `multi_agent_social_nav`, `multi_agent_hssd_spot_human`
- **Collaborative**: `multi_agent_tidy_house`

## 🛠 Python API

### Basic Benchmarking
```python
from easim.benchmark.benchmark import HabitatBenchmark
from easim.agents.sample import SampleAgent

# Initialize benchmark and agent
benchmark = HabitatBenchmark("objectnav_hm3d")
agent = SampleAgent()

# Run evaluation
metrics = benchmark.evaluate(agent, num_episodes=10)
print(f"Success Rate: {metrics['success']:.3f}")

# With video recording
metrics = benchmark.evaluate(agent, num_episodes=5, record_video=True)
```

### Video Recording
```python
from easim.utils.video_recorder import VideoRecorder
from pathlib import Path

# Setup video directory
video_dir = VideoRecorder.setup_video_directory("my_task")

# Record single episode with video
env = benchmark._env
metrics = VideoRecorder.record_episode_with_video(
    env, agent, episode_num=0, video_dir=video_dir
)

# Record without video (faster)
metrics = VideoRecorder.record_episode_no_video(env, agent)
```

### Custom Agent Integration
```python
from habitat import Agent

class MyAgent(Agent):
    def reset(self):
        pass
    
    def act(self, observations):
        # Your agent logic here
        return {"action": "move_forward"}

# Use with benchmark
agent = MyAgent()
benchmark = HabitatBenchmark("pointnav_hm3d")
results = benchmark.evaluate(agent, num_episodes=10)
```

## 📁 Project Structure

```
EmbodiedAgentSim/
├── easim/
│   ├── agents/                 # Agent implementations
│   │   └── sample.py          # Sample random agent
│   ├── benchmark/             # Benchmarking framework
│   │   └── benchmark.py       # HabitatBenchmark class
│   ├── cli/                   # Command line interface
│   │   ├── commands.py        # Command handlers
│   │   └── parser.py          # Argument parsing
│   ├── evaluation/            # Evaluation metrics and analysis
│   ├── examples/              # Usage examples
│   │   ├── interactive.py     # Interactive demo
│   │   └── video.py          # Video recording example
│   └── utils/                 # Utilities and constants
│       ├── constants.py       # Benchmark configs and paths
│       ├── video_recorder.py  # Video recording functionality
│       └── habitat_utils.py   # Habitat-lab setup
├── data/                      # Auto-created data directory
│   └── output/               # Benchmark results and videos
└── habitat-lab/              # Habitat-lab submodule
```

## 🎮 Command Reference

### Core Commands
| Command | Description | Example |
|---------|-------------|---------|
| `list-tasks` | Show all available benchmark tasks | `easim list-tasks` |
| `interactive` | Run interactive navigation demo | `easim interactive` |
| `benchmark` | Run agent benchmarking | `easim benchmark --task objectnav_hm3d` |

### Benchmark Options
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--task` | Benchmark task to run | `objectnav_hm3d` | `--task pointnav_mp3d` |
| `--episodes` | Number of episodes | `10` | `--episodes 5` |
| `--video` | Enable video recording | `False` | `--video` |
| `--agent` | Agent type | `sample` | `--agent custom` |

## 📊 Output and Results

### Benchmark Metrics
Results include standard Habitat-Lab metrics:
- **Success Rate**: Episode completion percentage
- **SPL**: Success weighted by Path Length
- **Distance to Goal**: Final distance to target
- **Episode Length**: Steps taken per episode

### Video Output
When `--video` is enabled:
- Videos saved to: `data/output/videos/{task_name}/run_001/`
- Format: `episode_001.mp4`, `episode_002.mp4`, etc.
- Automatic run numbering prevents overwrites

## 🔧 Advanced Usage

### Custom Benchmark Configuration
```python
# Access all available benchmarks
from easim.utils.constants import BENCHMARK_CONFIG

print(f"Available tasks: {len(BENCHMARK_CONFIG)}")
for task, config_path in BENCHMARK_CONFIG.items():
    print(f"{task}: {config_path}")
```

### Batch Evaluation
```bash
# Run multiple tasks
for task in objectnav_hm3d pointnav_mp3d imagenav_gibson; do
    easim benchmark --task $task --episodes 10 --video
done
```

## 📚 Dataset Requirements

The framework requires Habitat-Lab datasets. Refer to [Habitat-Lab documentation](https://github.com/facebookresearch/habitat-lab) for dataset setup:

- **Scene Datasets**: HM3D, MP3D, Gibson, HSSD, ProcTHOR
- **Task Datasets**: ObjectNav, PointNav, ImageNav, Rearrangement
- **Installation**: Follow Habitat-Lab dataset installation guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built on top of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) by Facebook AI Research.