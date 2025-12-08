# RL-AutonomousDriving-CS-272

Reinforcement Learning project for autonomous driving using HighwayEnv environments with custom construction zones.

**Course:** CS 272  
**Team:** Mitansh Gor, Henry Ha, John Yun

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Training Scripts](#training-scripts)
- [Custom Environment](#custom-environment)
- [Documentation](#documentation)
- [Tests](#tests)
- [Requirements](#requirements)

---

## Overview

This project implements deep reinforcement learning (DRL) agents for autonomous driving in simulated highway environments with construction zones. The project features:

1. **Custom Highway Environment:** `HighwayWithObstaclesEnv` - A highway environment with static obstacles and construction zones marked by traffic cones
2. **DRL Training:** Train agents using DQN, PPO, and SAC algorithms
3. **Construction Zone Navigation:** Agents learn to navigate through construction zones while maintaining safety and efficiency

The project extends the [HighwayEnv](https://highway-env.farama.org/) framework built on Gymnasium.

---

## Project Structure

```
RL-AutonomousDriving-CS-272/
├── highway_env/           # Extended HighwayEnv package
│   ├── envs/             # Environment implementations
│   │   ├── highway_with_obstacles_env.py  # Custom construction zone environment
│   │   ├── highway_env.py
│   │   ├── merge_env.py
│   │   ├── intersection_env.py
│   │   └── common/       # Common utilities
│   ├── road/             # Road infrastructure
│   ├── vehicle/          # Vehicle dynamics and behavior
│   └── utils.py
├── scripts/              # Training scripts
│   ├── train_dqn.py      # DQN training script
│   ├── train_ppo.py      # PPO training script
│   ├── train_sac.py      # SAC training script
│   ├── train_all_models.py  # Train all models for comparison
│   ├── train_rl.py       # Template for custom RL setup
│   └── README.md         # Training scripts documentation
├── examples/             # Example scripts
│   └── example_usage.py  # Usage examples
├── config/               # Configuration files
│   └── env_config.json   # Environment configuration
├── docs/                 # Documentation
│   ├── ConstructionZoneEnv.md
│   └── requierements.md
├── highway_dqn/          # Pre-trained models
│   ├── model.zip
│   └── videos/
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

**Note:** The following directories are created automatically during training:
- `models/` - Trained RL models
- `logs/` - Training logs and TensorBoard data
- `plots/` - Generated plots and visualizations
- `results/` - Evaluation results and metrics


---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Step 1: Clone or Navigate to Project

```bash
# If cloning from git
git clone https://github.com/MitanshGor/RL-AutonomousDriving-CS-272.git
cd RL-AutonomousDriving-CS-272

# Or navigate to existing project directory
cd /path/to/RL-AutonomousDriving-CS-272
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "import gymnasium; import highway_env; print('✓ Installation successful!')"
```

---

## Quick Start

### Basic Environment Usage

```python
import gymnasium as gym
import highway_env  # Registers custom environments

# Create environment
env = gym.make('highway-with-obstacles-v0', render_mode='rgb_array')

# Configure environment
env.unwrapped.config.update({
    "obstacles_count": 4,                    # Number of static obstacles
    "obstacle_spacing": 20,                  # Min spacing between obstacles [m]
    "vehicles_count": 50,                    # Number of other vehicles
    "construction_zones_count": 2,           # Number of construction zones
    "construction_zone_length": 150,         # Length of each zone [m]
    "construction_zone_side": "random",      # "left", "right", or "random"
    "construction_zone_lanes": 2,            # Lanes the zone takes up
    "construction_cone_spacing": 5,          # Distance between cones [m]
})

# Reset and run
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

---

### Model training
when you create your first model make sure to save the model after training
you can save it to a highway_<model_name>/model directory

```python
# The training script uses the HighwayWithObstaclesEnv with construction zones
model.learn(total_timesteps=int(2e4))
model.save("highway_dqn/model")
```

### Model training Cont.
run your training model.
```bash
# Train a DQN model on highway with obstacles (default construction zone environment)
python scripts/train_dqn.py
```


## Usage

### Custom Environment: HighwayWithObstaclesEnv

The environment features:
- **Static Obstacles:** Configurable number of obstacles on the highway
- **Construction Zones:** Dynamic construction zones marked with traffic cones
- **Flexible Configuration:** Customize zone length, side, lane coverage, and cone spacing
- **Continuous Action Space:** Ego-vehicle takes continous actions with a range of values for steering direction and throttle position

### Available Base Environments

The project also supports standard HighwayEnv environments:
- **`highway-v0`** - Standard highway driving
- **`merge-v0`** - Highway merge scenario
- **`intersection-v0`** - Intersection navigation
- **`highway-with-obstacles-v0`** - Custom construction zone environment (this project)

### Configuration File

The `config/env_config.json` file contains detailed configuration for action spaces, rewards, and episode parameters. This configuration is used for reference.

---

## Custom Environment

### HighwayWithObstaclesEnv Features

The custom `HighwayWithObstaclesEnv` extends the standard HighwayEnv with:

1. **Static Obstacles:** Randomly placed obstacles on the highway
2. **Construction Zones:** Marked areas with traffic cones that create lane restrictions
3. **Dynamic Configuration:** Adjust zone count, length, affected lanes, and cone spacing
4. **Realistic Constraints:** Vehicles must navigate around both static obstacles and construction zones

### Construction Zone Configuration

```python
config = {
    "construction_zones_count": 2,      # Number of zones
    "construction_zone_length": 100,    # Length in meters
    "construction_zone_side": "random", # "left", "right", or "random"
    "construction_zone_lanes": 2,       # Number of lanes affected
    "construction_cone_spacing": 5,     # Distance between cones [m]
}
```

### Obstacle Configuration

```python
config = {
    "obstacles_count": 10,              # Number of static obstacles
    "obstacle_spacing": 20,             # Minimum spacing [m]
    "obstacle_on_lanes": None,          # None for all lanes, or list of lane indices
}
```

See [docs/ConstructionZoneEnv.md](docs/ConstructionZoneEnv.md) for detailed specifications.

---

## Training Scripts

### Available Training Scripts

The project includes three RL algorithm implementations using Stable-Baselines3:

1. **`scripts/train_dqn.py`** - Deep Q-Network (discrete actions)

more to be created...

### Model Output

Trained models and videos are saved to:
- `highway_dqn/model.zip` - Trained DQN model
- `highway_dqn/videos/` - Recorded episode videos

For other algorithms, models are typically saved in:
- `models/{algorithm}_{env}/` - Model checkpoints
- `logs/` - TensorBoard training logs

### Viewing Training Progress

```bash
# If TensorBoard logging is enabled
tensorboard --logdir logs/
```

### Custom RL Setup

For custom RL implementations, use `scripts/train_rl.py` as a template. See [scripts/README.md](scripts/README.md) for detailed documentation.

---

## Documentation

- **[docs/ConstructionZoneEnv.md](docs/ConstructionZoneEnv.md)** - Custom environment specifications
- **[docs/requierements.md](docs/requierements.md)** - Project requirements and guidelines
- **[config/env_config.json](config/env_config.json)** - Environment configuration reference

---


## Tests

The `tests/` directory contains automated tests to verify the correct behavior of the environments, utilities, and core components. Tests are organized by module and cover:

- **Environment API compliance** (e.g., `test_gym.py`)
- **Step and reset logic** for all environments
- **Utility functions** (e.g., collision detection)
- **Vehicle, road, and rendering logic**

### Directory Structure

```
tests/
├── envs/         # Tests for environment logic and Gym API
│   ├── test_gym.py
│   ├── test_actions.py
│   ├── test_env_preprocessors.py
│   └── ...
├── graphics/     # Rendering and visualization tests
├── road/         # Road and lane logic tests
├── vehicle/      # Vehicle dynamics and behavior tests
├── test_utils.py # Utility function tests
└── __init__.py
```

### Running the Tests

You can run all tests using `pytest` from the project root:

```bash
pytest tests
```

Or run a specific test file:

```bash
pytest tests/envs/test_gym.py
```

If all tests pass, your environment and codebase are working as expected.

---
## Requirements

### Python Packages

Main dependencies (see `requirements.txt` for complete list):

```
gymnasium>=0.29.0      # RL environment framework
highway-env>=1.8.0     # Highway driving environments
numpy>=1.24.0          # Numerical computing
stable-baselines3>=2.0.0  # RL algorithms (DQN, PPO, SAC)
tensorboard>=2.10.0    # Training visualization
tqdm>=4.60.0           # Progress bars
rich>=10.0.0           # Terminal formatting
```

### Installation

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Verify highway-env is installed
python -c "import highway_env; print('highway-env installed')"

# Check gymnasium version
python -c "import gymnasium; print(gymnasium.__version__)"

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### Environment Registration

If the custom environment is not found:

perharps the environment was not registered (but should exist in the __init__.py file within the highway_env directory)
```python
import gymnasium as gym
import highway_env  # This import registers the environment

gym.register(
    id='highway-with-obstacles-v0',
    entry_point='highway_env.envs:HighwayWithObstaclesEnv',
)
```

### Rendering Issues

For rendering issues on Windows:

```bash
# Install pygame if using 'human' render mode
pip install pygame
```

---

## Contributing

This is a course project for CS 272. For questions or collaboration, contact the team members.

---

## Acknowledgments

This project is built on top of:
- **[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)** - Farama Foundation
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** - DLR-RM
- **[Gymnasium](https://gymnasium.farama.org/)** - Farama Foundation

---

## License

MIT

---

## Contact

**Team Members:**
- Mitansh Gor
- Henry Ha
- John Yun

**Course:** CS 272  
**Institution:** [Your Institution Name]

**GitHub Repository:** [RL-AutonomousDriving-CS-272](https://github.com/MitanshGor/RL-AutonomousDriving-CS-272)

