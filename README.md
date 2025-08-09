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
# Object Navigation
easim benchmark --task objectnav_hm3d --episodes 10
easim benchmark --task objectnav_mp3d --episodes 10 --record

# Embodied Question Answering
easim benchmark --task eqa_hm3d --episodes 10
easim benchmark --task eqa_rgbonly_hm3d --episodes 5 --record

# Express-Bench (Rich trajectory data - 2044 episodes)
easim benchmark --task eqa_hm3d_express --episodes 20 --record
easim benchmark --task eqa_rgbonly_hm3d_express --episodes 15
```

### Interactive Demo
```bash
# Run interactive navigation demo
easim interactive
```

## 📋 Available Benchmark Categories

### Tuned and Ready Tasks

#### Object Navigation
- **ObjectNav HM3D**: `objectnav_hm3d`, `objectnav_hm3d_with_semantic`
- **ObjectNav MP3D**: `objectnav_mp3d`, `objectnav_mp3d_with_semantic`

#### Embodied Question Answering
- **Standard EQA**: `eqa_hm3d`, `eqa_rgbonly_hm3d` (500 episodes)
- **Express-Bench**: `eqa_hm3d_express`, `eqa_rgbonly_hm3d_express` (2044 episodes with rich trajectory data)

### Additional Available Tasks (Not Yet Tuned)
- Point Navigation, Image Navigation, VLN, Rearrangement, Multi-Agent tasks
- Use `easim list-tasks` to see all 58+ available benchmarks

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
| `--record` | Enable video recording | `False` | `--record` |
| `--agent` | Agent type | `sample` | `--agent custom` |

## 📊 Output and Results

### Benchmark Metrics
Results include standard Habitat-Lab metrics:
- **Success Rate**: Episode completion percentage
- **SPL**: Success weighted by Path Length
- **Distance to Goal**: Final distance to target
- **Episode Length**: Steps taken per episode

### Video Output
When `--record` is enabled:
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
# Run multiple ObjectNav and EQA tasks
for task in objectnav_hm3d objectnav_mp3d eqa_hm3d eqa_hm3d_express; do
    easim benchmark --task $task --episodes 10 --record
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