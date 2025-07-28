# EmbodiedAgentSim

A modular framework for embodied AI simulation supporting Object Navigation, Vision-Language Navigation, and Embodied Question Answering.

## 📦 Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/EmbodiedAgentSim.git
cd EmbodiedAgentSim

# Create environment  
conda env create -f environment.yml
conda activate habitat

# Install package
pip install -e .
```

## 🚀 Commands

### Simulator Testing
```bash
# Test MP3D simulator
easim simulator --dataset MP3D

# Test HM3D simulator  
easim simulator --dataset HM3D --scene-path /path/to/scene.glb
```



### Video Recording
```bash
# Record random navigation
easim record --dataset MP3D --max-steps 100 --video-name exploration.mp4

# With custom settings
easim record --dataset HM3D --max-steps 50 --fps 24 --save-frames

# Custom output directory
easim record --dataset MP3D --output-dir ./videos --video-name test.mp4
```

### Interactive Control
```bash
# Interactive navigation (W/A/D keys, ESC to quit)
easim interactive --dataset MP3D

# With custom scene
easim interactive --dataset HM3D --scene-path /path/to/scene.glb
```

### Run Examples
```bash
# Video recording examples
python -m easim.examples.video_recording

# General use cases
python -m easim.examples.use_cases

# R2R dataset examples  
python -m easim.examples.r2r
```

### Python API
```python
from easim.core.simulator import CoreSimulator, SimulatorConfig

# Basic usage
config = SimulatorConfig(dataset_type="MP3D")
simulator = CoreSimulator(config)

# Get observations and take actions
obs = simulator.get_observations()
new_obs = simulator.step("move_forward")

# Record video
result = simulator.record_random_navigation("output.mp4", max_steps=100)

# Interactive mode
simulator.run_interactive()

simulator.close()
```

## 📁 Project Structure

```
EmbodiedAgentSim/
├── easim/
│   ├── core/                    # Core simulation (actions, sensors, agents, simulator)
│   ├── tasks/                   # Task implementations (ObjectNav, EQA, etc.)
│   ├── datasets/               # Dataset loaders (HM3D, MP3D, R2R)
│   ├── examples/               # Example scripts
│   └── utils/                  # Constants and utilities
├── data/                       # Auto-created data directory
└── requirements.txt            # Dependencies
```

## 🎯 Quick Start

1. **Test simulator**: `easim simulator --dataset MP3D`
2. **Record video**: `easim record --dataset MP3D --max-steps 50 --video-name test.mp4`
3. **Interactive mode**: `easim interactive --dataset MP3D`
4. **Run examples**: `python -m easim.examples.video_recording`

## 📚 Dataset Support

- **Scene Datasets**: MP3D, HM3D
- **Task Datasets**: ObjectNav, EQA, R2R VLN
- **Auto-download**: Place datasets in `data/` directory

## 🔧 Common Options

| Option | Description | Default | Commands |
|--------|-------------|---------|----------|
| `--dataset` | MP3D/HM3D for scenes, hm3d/mp3d/r2r for tasks | MP3D | All |
| `--task-type` | objectnav, eqa, vln | - | task, evaluate |
| `--episodes` | Number of episodes to run | 10 | task, evaluate |
| `--agent` | random, forward | random | task, evaluate |
| `--split` | Dataset split (train/val/test) | val | task, evaluate |
| `--scene-path` | Custom scene file | Auto | simulator, interactive |
| `--max-steps` | Navigation steps | 100 | record |
| `--fps` | Video frame rate | 30 | record |
| `--save-frames` | Save frame images | False | record |
| `--record-video` | Record task execution | False | task |
| `--output-dir` | Output directory | data/output | record, task |
| `--output-file` | Save results JSON | None | evaluate |

## 📄 License

MIT License - see [LICENSE](LICENSE) file.