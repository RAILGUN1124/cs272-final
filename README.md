# CS 272 Final Project: Deep Reinforcement Learning for Autonomous Driving

This project explores deep reinforcement learning (DRL) approaches for autonomous driving across multiple highway scenarios using the HighwayEnv framework. The project includes implementations of PPO agents trained on both standard and custom highway environments with various observation types.

## 📋 Overview

This repository contains five distinct autonomous driving experiments:

1. **Highway Environment** - Standard highway driving with lane changes and traffic navigation
2. **Intersection Environment** - Complex intersection navigation with crossing traffic
3. **Merge Environment** - Highway merging scenarios with dense traffic
4. **Custom Environment (2a)** - Narrow lane environment with safe lane-changing focus
5. **Custom Environment (2b)** - Highway with construction zones and static obstacles

Each experiment compares different observation types (Grayscale vision vs. LiDAR sensors) to analyze their effectiveness in training autonomous driving agents.

---

## 🗂️ Project Structure

```
cs272-final/
├── Highway/                    # Standard highway environment experiments
│   ├── train_highway_grayscale.py
│   ├── train_highway_lidar.py
│   ├── evaluate_and_plot.py
│   ├── models/                # Trained models (.zip)
│   ├── logs/                  # TensorBoard logs
│   ├── training_data/         # Training metrics (JSON)
│   ├── evaluation_data/       # Evaluation results (JSON)
│   └── plots/                 # Generated visualizations
│
├── Intersection/              # Intersection environment experiments
│   ├── train_intersection_grayscale.py
│   ├── train_intersection_lidar.py
│   ├── evaluate_and_plot.py
│   └── [same structure as Highway/]
│
├── Merge/                     # Merge environment experiments
│   ├── train_merge_grayscale.py
│   ├── train_merge_lidar.py
│   ├── evaluate_and_plot.py
│   └── [same structure as Highway/]
│
├── 2a)CustomEnv/              # Our Narrow lane custom environment
│   ├── narrow_lane_env.py    # Custom environment definition
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_and_plot.py
│   └── [logs, models, plots directories]
│
└── 2b)CustomEnv/              # Group 10's Construction zone custom environment
    ├── highway_env/          # Extended HighwayEnv package
    │   └── envs/
    │       └── exit_env.py   # Custom construction zone environment
    ├── scripts/
    │   ├── train.py
    │   ├── train_dqn.py
    │   ├── evaluate.py
    │   └── generate_plots.py
    ├── config/               # Environment configurations
    ├── docs/                 # Detailed documentation
    └── [logs, models, results directories]
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install gymnasium highway-env stable-baselines3 matplotlib seaborn numpy tqdm
```

### Running Experiments

Each environment folder contains training and evaluation scripts:

**Standard Environments (Highway, Intersection, Merge):**
```bash
# Navigate to desired environment
cd Highway/  # or Intersection/ or Merge/

# Train with grayscale observation
python train_highway_grayscale.py  # Takes ~30-60 min

# Train with LiDAR observation
python train_highway_lidar.py

# Evaluate and generate plots
python evaluate_and_plot.py
```

**Custom Environment 2a (Narrow Lane):**
```bash
cd 2a)CustomEnv/

# Train PPO agent
python train.py

# Evaluate trained model
python evaluate.py

# Generate plots
python evaluate_and_plot.py
```

**Custom Environment 2b (Construction Zone):**
```bash
cd 2b)CustomEnv/

# Train using different algorithms
python scripts/train.py      # PPO training
python scripts/train_dqn.py  # DQN training

# Evaluate models
python scripts/evaluate.py

# Generate visualizations
python scripts/generate_plots.py
```

---

## 🎯 Experiments Summary

### Standard Environments

| Environment | Training Steps | Observation Types | Key Features |
|-------------|---------------|-------------------|--------------|
| **Highway** | 100,000 | Grayscale (128×64×4)<br>LiDAR (64 cells) | Multi-lane driving, overtaking, collision avoidance |
| **Intersection** | 500,000 | Grayscale (128×64×4)<br>LiDAR (64 cells) | Crossing traffic, right-of-way, complex decision-making |
| **Merge** | 500,000 | Grayscale (128×64×4)<br>LiDAR (64 cells) | Highway merging, gap acceptance, speed matching |

### Custom Environments

#### 2a) Narrow Lane Environment
- **Focus:** Safe lane-changing in constrained two-lane highway
- **Traffic:** 10 vehicles with disabled lane changes
- **Observation:** Kinematics (5 nearest vehicles)
- **Actions:** Discrete (lane left/right, faster/slower, idle)
- **Reward:** Survival (0.1) + Lane changes (0.2)
- **Challenge:** Collision avoidance with stationary-lane traffic

#### 2b) Construction Zone Environment
- **Focus:** Navigation through highway construction zones
- **Features:** Static obstacles, traffic cones, zone markers
- **Observation:** Customizable (Kinematics/Grayscale/LiDAR)
- **Actions:** Continuous (steering, acceleration)
- **Challenge:** Dynamic obstacle avoidance in constrained spaces
- **Algorithms:** DQN, PPO, SAC support

---

## 📊 Observation Types

### Grayscale Vision
- **Resolution:** 128×64 pixels
- **Stack:** 4 consecutive frames
- **Weights:** RGB→Gray [0.2989, 0.5870, 0.1140]
- **Advantages:** Rich spatial information, scene understanding
- **Network:** CNN-based feature extraction

### LiDAR Sensors
- **Cells:** 64 rangefinder readings
- **Advantages:** Precise distance measurements, computationally efficient
- **Network:** MLP-based processing

---

## 🤖 Model Architecture

All standard environment experiments use **PPO (Proximal Policy Optimization)** with:
- Multi-layer perceptron (MLP) policy for LiDAR observations
- Convolutional Neural Network (CNN) for grayscale observations
- Custom reward callbacks for training progress tracking
- Episode monitoring and logging
- TensorBoard integration

Custom environments support multiple algorithms:
- **PPO:** Stable, sample-efficient policy gradient method
- **DQN:** Value-based deep Q-learning

---

## 📈 Evaluation & Visualization

Each environment generates:
- **Learning curves:** Training progress over time
- **Violin plots:** Reward distribution across evaluation episodes
- **JSON metrics:** Detailed training and evaluation statistics
- **TensorBoard logs:** Real-time training monitoring

Access visualizations in the `plots/` directory of each environment.

---

## 📚 Documentation

Detailed documentation for each component:
- `Highway/README.md` - Highway environment details
- `Intersection/README.md` - Intersection environment details
- `Merge/README.md` - Merge environment details
- `2a)CustomEnv/README.md` - Narrow lane environment specification
- `2b)CustomEnv/README.md` - Construction zone environment guide
- `2b)CustomEnv/docs/` - Extended documentation for custom environment 2b

---

## 🔧 Configuration

Environment configurations can be modified in training scripts or config files:
- Observation types and parameters
- Reward function weights
- Episode duration and frequencies
- Vehicle counts and behavior
- Traffic density and speed ranges

See individual environment READMEs for specific configuration options.

---

## 📦 Dependencies

Core requirements:
- `gymnasium` - Environment interface
- `highway-env` - Highway driving environments
- `stable-baselines3` - RL algorithms (PPO, DQN, SAC)
- `torch` - Deep learning backend
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `numpy` - Numerical computing
- `tqdm` - Progress bars

See `requirements.txt` in each folder for complete dependencies.

---

## 📝 Citation

This project builds upon:
- [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) - Farama Foundation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - DLR-RM
- [Gymnasium](https://gymnasium.farama.org/) - Farama Foundation
