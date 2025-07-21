# EmbodiedAgentSim

A comprehensive, modular framework for embodied AI simulation supporting multiple tasks and datasets including Object Navigation, Vision-Language Navigation, and Embodied Question Answering.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Habitat-Sim](https://img.shields.io/badge/Habitat--Sim-0.3.0+-green.svg)](https://github.com/facebookresearch/habitat-sim)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Features

### 🎯 **Multi-Task Support**
- **Object Navigation (ObjectNav)** - Navigate to find specific objects
- **Vision-Language Navigation (VLN)** - Follow natural language instructions  
- **Embodied Question Answering (EQA)** - Answer questions through exploration
- **Point Navigation (PointNav)** - Navigate to specific coordinates

### 🗂️ **Multi-Dataset Support**
- **HM3D** - Habitat-Matterport 3D Dataset
- **MP3D** - Matterport3D scenes
- **R2R** - Room-to-Room VLN dataset

### 🏗️ **Modular Architecture**
- **Clean separation** of tasks, datasets, and evaluation
- **Extensible design** - easy to add new tasks and datasets
- **Proper inheritance** following Habitat framework patterns
- **Configuration-driven** setup with dataclass configs

### 📊 **Comprehensive Evaluation**
- **Standard metrics** for each task (Success Rate, SPL, Navigation Error, Accuracy)
- **Built-in evaluation framework** with detailed reporting
- **Episode management** with proper data structures
- **Metric visualization** and analysis tools

### 🎥 **Recording & Visualization**
- **Automatic video recording** from navigation episodes
- **Interactive control** with pygame-based interface
- **Frame-by-frame analysis** and trajectory visualization
- **Custom recording strategies** and policies

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/EmbodiedAgentSim.git
cd EmbodiedAgentSim

# Create conda environment
conda env create -f environment.yml
conda activate habitat

# Install package
pip install -e .

# Download test data
easim dataset download habitat_test_scenes
```

### Development Setup

```bash
# Install with all optional dependencies
pip install -e .[full]

# Or install specific components
pip install -e .[interactive]  # For pygame support
pip install -e .[habitat]      # For habitat-lab integration
```

## 🎯 Quick Start

### Basic Simulator Test
```bash
# Test MP3D simulator
easim simulator --dataset MP3D

# Test HM3D simulator  
easim simulator --dataset HM3D --list-scenes
```

### Run Tasks

```bash
# Object Navigation on HM3D
easim task --task-type objectnav --dataset hm3d --episodes 10

# R2R Vision-Language Navigation
easim task --task-type vln --dataset r2r --split val_seen --episodes 5

# Embodied Question Answering
easim task --task-type eqa --dataset mp3d --episodes 5
```

### Evaluation

```bash
# Evaluate random agent on ObjectNav
easim evaluate --task-type objectnav --dataset hm3d --agent random --episodes 50

# Evaluate on R2R with specific metrics
easim evaluate --task-type vln --dataset r2r --agent forward --episodes 100 --output-file results.json
```

### Recording & Interaction

```bash
# Record navigation video
easim record --dataset MP3D --task-type objectnav --max-steps 100 --video-name exploration.mp4

# Interactive control
easim interactive --dataset HM3D --task-type navigation
# Controls: W/↑=Forward, A/←=Left, D/→=Right, ESC=Quit
```

## 🏗️ Architecture

### Project Structure

```
EmbodiedAgentSim/
├── easim/
│   ├── core/                    # Core simulation components
│   │   ├── actions.py          # Action space definitions
│   │   ├── sensors.py          # Sensor configurations
│   │   ├── agents.py           # Agent configurations
│   │   └── simulator.py        # Core simulator
│   ├── tasks/                   # Task implementations
│   │   ├── base/               # Base framework
│   │   │   ├── task.py         # Base task class
│   │   │   ├── episode.py      # Episode data structures
│   │   │   ├── metrics.py      # Evaluation metrics
│   │   │   └── configs.py      # Configuration dataclasses
│   │   ├── nav/                # Navigation tasks
│   │   │   └── objectnav/      # Object Navigation
│   │   ├── vln/                # Vision-Language Navigation
│   │   │   └── r2r/            # Room-to-Room dataset
│   │   └── eqa/                # Embodied Question Answering
│   │       └── mp3deqa/        # MP3D EQA dataset
│   ├── datasets/               # Dataset loaders
│   │   ├── base/               # Base dataset classes
│   │   ├── hm3d/               # HM3D dataset support
│   │   ├── mp3d/               # MP3D dataset support
│   │   └── r2r/                # R2R dataset support
│   ├── sim/                    # Legacy simulation components
│   └── utils/                  # Utilities and constants
├── data/                       # Data directory
│   ├── scene_datasets/         # Scene files (MP3D, HM3D)
│   ├── datasets/               # Task datasets
│   └── output/                 # Generated outputs
└── examples/                   # Example scripts
```

### Core Components

#### Action Spaces
```python
from easim.core.actions import VLNActionSpace, EQAActionSpace

# VLN action space with stop action
vln_actions = VLNActionSpace(
    forward_step=0.25,
    turn_angle=15.0
)

# EQA action space with look and answer actions
eqa_actions = EQAActionSpace(
    forward_step=0.25,
    turn_angle=30.0,
    answer_vocab_size=3000
)
```

#### Sensor Suites
```python
from easim.core.sensors import ObjectNavSensorSuite, VLNSensorSuite

# ObjectNav sensors (RGB, depth, GPS, compass, object goal)
objectnav_sensors = ObjectNavSensorSuite(
    resolution=(256, 256),
    camera_height=1.25
)

# VLN sensors (RGB, depth, instruction)
vln_sensors = VLNSensorSuite(
    resolution=(224, 224),
    max_instruction_length=512
)
```

#### Agents
```python
from easim.core.agents import ObjectNavAgent, VLNAgent

# Create ObjectNav agent
agent = ObjectNavAgent(
    forward_step=0.25,
    turn_angle=30.0,
    resolution=(256, 256)
)

# Create VLN agent  
vln_agent = VLNAgent(
    forward_step=0.25,
    turn_angle=15.0,
    resolution=(224, 224)
)
```

## 🎯 Tasks

### Object Navigation

Navigate indoor environments to find specific object categories.

```python
from easim.tasks.nav.objectnav.task import create_objectnav_task

# Create ObjectNav task
task = create_objectnav_task(
    dataset_name="hm3d",
    split="val", 
    max_episodes=100
)

# Simple random agent
def random_agent(observations, step):
    import random
    return random.choice(["move_forward", "turn_left", "turn_right"])

# Run evaluation
from easim.tasks.base.task import TaskRunner
runner = TaskRunner(task)
results = runner.run_evaluation(10, random_agent)

print(f"Success Rate: {results['metrics']['success_rate']:.3f}")
print(f"SPL: {results['metrics']['spl']:.3f}")
```

**Supported Datasets**: HM3D, MP3D  
**Metrics**: Success Rate, SPL, Path Length, Distance to Goal  
**Object Categories**: 21 categories for HM3D, 16 for MP3D

### Vision-Language Navigation

Navigate by following natural language instructions.

```python
from easim.tasks.vln.r2r.task import create_r2r_task

# Create R2R task
task = create_r2r_task(
    split="val_seen",
    max_episodes=50
)

# Instruction-following agent (simplified)
def instruction_agent(observations, step):
    instruction = observations.get('instruction', '')
    
    # Simple keyword-based navigation
    if any(word in instruction.lower() for word in ['left', 'turn']):
        return 'turn_left'
    elif any(word in instruction.lower() for word in ['right']):
        return 'turn_right' 
    elif 'stop' in instruction.lower() or step > 100:
        return 'stop'
    else:
        return 'move_forward'

# Evaluate
runner = TaskRunner(task)
results = runner.run_evaluation(5, instruction_agent)
```

**Dataset**: R2R (Room-to-Room)  
**Metrics**: Success Rate, SPL, Navigation Error  
**Instructions**: Natural language navigation instructions

### Embodied Question Answering

Answer questions about the environment through exploration.

```python
from easim.tasks.eqa.mp3deqa.task import create_mp3d_eqa_task

# Create EQA task
task = create_mp3d_eqa_task(
    split="val",
    max_episodes=20
)

# Simple EQA agent
def eqa_agent(observations, step):
    question = observations.get('question', '')
    
    # Explore for a bit, then answer
    if step < 20:
        actions = ["move_forward", "turn_left", "turn_right", "look_up", "look_down"]
        import random
        return random.choice(actions)
    else:
        return "answer"  # Give answer after exploration

# Evaluate
runner = TaskRunner(task)
results = runner.run_evaluation(3, eqa_agent)
```

**Dataset**: MP3D EQA  
**Metrics**: Answer Accuracy, Mean Reciprocal Rank  
**Question Types**: Existence, counting, spatial, color, object category

## 📊 Evaluation & Metrics

### Standard Metrics

```python
from easim.tasks.base.metrics import SuccessRate, SPL, PathLength

# Create custom metric calculator
metrics = [
    SuccessRate(),
    SPL(), 
    PathLength()
]

from easim.tasks.base.metrics import MetricCalculator
calculator = MetricCalculator(metrics)

# Use in task evaluation
results = calculator.compute_final_metrics()
for name, result in results.items():
    print(f"{name}: {result.value:.4f}")
```

### Task-Specific Metrics

- **Navigation**: Success Rate, SPL, Path Length, Steps
- **VLN**: Navigation Error, Oracle Success  
- **EQA**: Answer Accuracy, Type-specific Accuracy

### Batch Evaluation

```bash
# Evaluate multiple episodes with detailed metrics
easim evaluate \
  --task-type objectnav \
  --dataset hm3d \
  --episodes 500 \
  --agent random \
  --output-file objectnav_results.json \
  --metrics success_rate spl path_length
```

## 🗂️ Datasets

### Downloading Datasets

```bash
# Download scene datasets
easim dataset download habitat_test_scenes
easim dataset download mp3d_example

# Download task datasets  
easim dataset download r2r

# List available datasets
easim dataset list --type scenes
easim dataset list --type tasks
```

### Dataset Support

| Dataset | Task | Scenes | Episodes | Status |
|---------|------|--------|----------|---------|
| HM3D | ObjectNav | 1000+ | 100k+ | ✅ Full |
| MP3D | ObjectNav | 90 | 70k+ | ✅ Full |
| MP3D | EQA | 90 | 50k+ | ✅ Full |
| R2R | VLN | 90 | 20k+ | ✅ Full |

### Custom Datasets

```python
from easim.datasets.base.dataset import BaseDataset, DatasetConfig
from easim.tasks.base.episode import ObjectNavEpisode

class MyDataset(BaseDataset):
    def load_episodes(self):
        # Load your episodes
        episodes = []
        # ... your loading logic
        self.episodes = episodes
        self._update_episode_dict()

# Use custom dataset
config = DatasetConfig(name="my_dataset", split="train")
dataset = MyDataset(config)
```

## 🎨 Advanced Usage

### Custom Tasks

```python
from easim.tasks.base.task import NavigationTask
from easim.tasks.base.configs import NavigationConfig

class MyNavigationTask(NavigationTask):
    def _check_success(self, episode, agent_position, observations):
        # Define custom success criteria
        return self._get_distance_to_goal(episode, agent_position) < 2.0
    
    def _calculate_reward(self, episode, observations, action, success):
        # Define custom reward function
        reward = super()._calculate_reward(episode, observations, action, success)
        
        # Add custom reward shaping
        if 'custom_sensor' in observations:
            reward += 0.1
        
        return reward

# Use custom task
config = NavigationConfig(max_episode_steps=300)
task = MyNavigationTask(config, simulator, episodes)
```

### Custom Agents

```python
from easim.core.agents import BaseAgent
from easim.tasks.base.configs import AgentConfig

# Custom agent configuration
config = AgentConfig(
    action_space_type="discrete_nav",
    sensor_suite_type="full_vision",
    height=1.8,  # Taller agent
    radius=0.3,  # Wider agent
    sensor_height=1.5
)

agent = BaseAgent(config)
```

### Configuration Management

```python
from easim.tasks.nav.objectnav.configs import ObjectNavConfig

# Detailed ObjectNav configuration
config = ObjectNavConfig(
    max_episode_steps=750,
    success_distance=1.5,
    target_object_categories=["chair", "table", "sofa"],
    view_success=True,
    step_penalty=-0.005,
    success_reward=15.0,
    collision_penalty=-0.2
)
```

## 🛠️ Development

### Adding New Tasks

1. **Create task directory**: `easim/tasks/mytask/`
2. **Implement task class**: Extend `BaseTask`
3. **Define configurations**: Create `configs.py` with dataclasses
4. **Add episodes**: Define episode structure
5. **Implement metrics**: Task-specific evaluation metrics
6. **Add CLI support**: Update `run.py`

```python
# Example new task structure
easim/tasks/mytask/
├── __init__.py
├── task.py          # Main task implementation
├── configs.py       # Configuration dataclasses
├── episode.py       # Episode data structures (if needed)
└── metrics.py       # Task-specific metrics (if needed)
```

### Adding New Datasets

1. **Create dataset directory**: `easim/datasets/mydataset/`
2. **Implement dataset class**: Extend `BaseDataset`  
3. **Define episode parser**: Convert data to episode format
4. **Add download utilities**: If public dataset
5. **Update constants**: Add paths and metadata

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Test specific components
python -m easim.tasks.nav.objectnav.task  # Run demo
python -m easim.datasets.r2r.dataset      # Test dataset loading
```

## 📚 API Reference

### Core Classes

- **`CoreSimulator`** - Main simulation engine
- **`BaseTask`** - Abstract task implementation
- **`BaseDataset`** - Abstract dataset loader
- **`BaseAgent`** - Agent configuration and management
- **`TaskRunner`** - Task execution and evaluation

### Task Classes

- **`ObjectNavigationTask`** - Object navigation implementation
- **`R2RTask`** - R2R VLN implementation  
- **`MP3DEQATask`** - MP3D EQA implementation

### Dataset Classes

- **`HM3DObjectNavDataset`** - HM3D ObjectNav episodes
- **`MP3DObjectNavDataset`** - MP3D ObjectNav episodes
- **`R2RDataset`** - R2R VLN episodes
- **`MP3DEQADataset`** - MP3D EQA episodes

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set data path
export EASIM_DATA_PATH=/path/to/data

# Optional: Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Optional: Enable debug logging
export EASIM_DEBUG=1
```

### Configuration Files

The framework uses dataclass-based configurations:

```python
from easim.tasks.base.configs import TaskConfig

config = TaskConfig(
    max_episode_steps=500,
    success_distance=1.0,
    step_penalty=-0.01,
    success_reward=10.0
)
```

## 🚨 Troubleshooting

### Common Issues

**Scene files not found**
```bash
# Check available scenes
easim info --list-scenes

# Download test scenes
easim dataset download habitat_test_scenes
```

**Import errors**
```bash
# Reinstall package
pip install -e .

# Check dependencies
easim info --check-deps
```

**Interactive mode not working**
```bash
# Install pygame
pip install pygame

# Or install with interactive support
pip install -e .[interactive]
```

**GPU/Memory issues**
- Use `--gpu-id 0` to specify GPU
- Reduce episode batch sizes
- Use lower resolution sensors: `resolution=(128, 128)`

### Performance Tips

- **Use GPU**: Set `gpu_device_id=0` in configs
- **Reduce resolution**: Use `(256, 256)` or lower for sensors  
- **Limit episodes**: Start with small episode counts
- **Cache scenes**: Enable scene caching for repeated use

## 📖 Examples

See the [`examples/`](examples/) directory for:

- **Basic navigation** - Simple random and forward agents
- **Custom agents** - Implementing your own navigation policies
- **Dataset loading** - Working with different datasets
- **Evaluation scripts** - Running comprehensive evaluations
- **Video recording** - Creating navigation videos

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Habitat-Sim](https://github.com/facebookresearch/habitat-sim)** - 3D simulation platform
- **[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)** - Task definitions and benchmarks
- **[Matterport3D](https://niessner.github.io/Matterport/)** - Scene dataset
- **[HM3D](https://aihabitat.org/datasets/hm3d/)** - Scene dataset  
- **[R2R Dataset](https://github.com/peteanderson80/Matterport3DSimulator)** - VLN benchmark

## 📚 Citation

If you use EmbodiedAgentSim in your research, please cite:

```bibtex
@software{embodied_agent_sim_2025,
  title={EmbodiedAgentSim: A Comprehensive Framework for Embodied AI Simulation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/EmbodiedAgentSim},
  note={Software framework for embodied AI research}
}
```

## 🔗 Links

- **Documentation**: [https://embodiedagentsim.readthedocs.io](https://embodiedagentsim.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/EmbodiedAgentSim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/EmbodiedAgentSim/discussions)
- **Paper**: [ArXiv Link](https://arxiv.org/abs/your-paper) (if available)

---

**Happy Simulating! 🤖🏠**