# 🚧 ConstructionZoneEnv — Custom Environment for HighwayEnv

**Course:** CS 272 — Final Project Discussion  
**Date:** November 11, 2025

---

## 🧠 Overview

`ConstructionZoneEnv` is a custom reinforcement learning environment built on top of the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) framework.  
It simulates a **multi-lane highway with an active construction zone**, where the self-driving agent must navigate safely through obstacles, barriers, and moving worker vehicles.

---

## 🎯 Objective

The agent’s goal is to:
- Drive through the construction area **without crashing**.
- **Stay in open lanes** (avoid closed or blocked lanes).
- Maintain **target speed** smoothly within the construction zone speed limit.

A successful episode ends when the ego vehicle **successfully completes the N sized threshold episode**.

---

## 🧩 Environment Setup

### **State (World)**
- Multi-lane highway with a temporary **construction zone**.
- Includes **cones**, **barriers**, and **slow-moving worker carts**.
- Standard traffic continues in unaffected lanes.

### **Agent**
- Self-driving car with **continuous steering and acceleration/brake control**.
- Action space: Continuous 2D vector `[acceleration, steering]`.

### **Episode Duration**
- ~20–40 seconds simulated (300–600 steps).

---

## 🧱 Terminal States

**✅ Successful:**  
- The vehicle successfully completes the N sized threshold episode, without collisions or rule violations.

**❌ Failure:**  
- Collision with another vehicle or barrier.  
- Entering a closed lane.  
- Not finishing within the episode duration.

---

## 🎮 Actions

The environment uses **continuous action space** for smooth, realistic vehicle control.

### **Action Space**
- **Action Type:** Continuous (2D vector)
- **Action Dimensions:**
  - `[acceleration, steering]`
    - `acceleration`: Continuous value for throttle/brake control
    - `steering`: Continuous value for lateral movement (lane transitions)

### **Action Configuration**

| Parameter | Min Threshold | Max Threshold | Description |
|-----------|---------------|---------------|-------------|
| **Acceleration** | `min_accel` (default: -1.0) | `max_accel` (default: +1.0) | Negative values = brake, positive values = throttle |
| **Steering** | `min_steering` (default: -1.0) | `max_steering` (default: +1.0) | Negative = transition left, positive = transition right, 0.0 = stay in lane |
| **Brake** | `min_brake` (default: -1.0) | `max_brake` (default: 0.0) | Applied when acceleration < 0 |

### **Lane Changes**
Lane changes are **continuous transitions** rather than discrete jumps:
- **Steering values** control the lateral movement smoothly between lanes
- Values closer to `±1.0` result in faster lane transitions
- Values near `0.0` maintain the current lane position
- The vehicle gradually transitions from the current lane to adjacent lanes based on steering input

---

## 💰 Reward Function

| Condition | Reward |
|------------|--------|
| Collision or closed lane | **−1.0** |
| Distance covered (percentage) | **+progress** (e.g., 0.73 if 73% covered) |
| Driving within speed limit | **+0.05 per step** |
| Speeding beyond limit | **−0.05 per step** |

**Goal:** Maximize smooth, safe progress through the construction zone.

---

## ⚙️ Constraints

- Avoid all **collisions** with vehicles, barriers, or cones.  
- Stay only in **open, valid lanes**.  
- Must maintain speed within **±5 mph** of the construction zone limit.  
- No off-road driving.

---

## 🔀 Randomization & Safety Rules

| Parameter | Range / Type |
|------------|---------------|
| Lanes count | 2–5 |
| Closed lane side | Left / Right |
| Barrier pattern | Solid |
| Traffic density | medium  |
| Worker cart count | 0–3 |
| Worker speed | 5–15 km/h |

**Safety Rules:**
- Closed lane = forbidden zone (instant penalty or fail).  
- Maintain safe following distance (time-to-collision > 1.2s).  
- Stay within the work-zone speed limit.

---

## 🧠 RL Training Plan

- **Algorithm:** TBD (Setup in `scripts/train_rl.py`)  
- **Curriculum Training:**  
  1. Static cones only.  
  2. Add moving worker carts.  
  3. Add dense traffic and fog.  

### **Evaluation Metrics**
- Success rate (safe exits).  
- Collision count per 100 episodes.  
- Mean episodic return.  
- Lane violation rate and jerk smoothness.

### **Training Setup**

**TODO: Setup your RL algorithm in `scripts/train_rl.py`**

The training script includes TODO comments for:
- RL algorithm initialization (PPO, SAC, DQN, etc.)
- Observation type configuration (LidarObservation, GrayscaleObservation)
- Hyperparameter tuning
- Model saving with experiment IDs
- Evaluation and plotting

---

## 🧰 Installation & Usage

### Project Setup

This custom environment is part of the RL-AutonomousDriving-CS-272 project. To use it:

```bash
# Install dependencies
pip install -r requirements.txt

# Use the environment wrapper
from src import HighwayEnvRunner

# Create environment with config
env = HighwayEnvRunner('highway', use_config=True)
```

### Custom Environment Implementation

To implement this custom environment in highway-env:

```bash
git clone https://github.com/Farama-Foundation/HighwayEnv.git
cd HighwayEnv
pip install -e .

# Add your custom env file
cp custom_envs/construction_zone_env.py highway_env/envs/

# Register and test
python -m highway_env.scripts.play --env highway-construction-zone-v0
```

**Note:** The `pip install -e .` command above is for installing the HighwayEnv package itself, not this project. This project does not require package installation - simply install dependencies with `pip install -r requirements.txt`.

### RL Algorithm Setup

**Ready-to-Use Training Scripts:**

The project includes training scripts for stable-baselines3 algorithms:
- `scripts/train_ppo.py` - PPO training
- `scripts/train_sac.py` - SAC training
- `scripts/train_dqn.py` - DQN training

**Usage:**
```bash
# Train PPO on custom environment
python scripts/train_ppo.py --env highway --timesteps 100000

# Train with specific observation type
python scripts/train_ppo.py --env highway --obs lidar --timesteps 100000
```

**Custom RL Setup:**

For custom implementations, use `scripts/train_rl.py` as a template. See [scripts/README.md](../scripts/README.md) for detailed documentation.

---

## 📊 Citation

If you use this environment, please cite:
> **HighwayEnv: A Flexible Gymnasium Environment for Autonomous Driving** — Farama Foundation (https://github.com/Farama-Foundation/HighwayEnv)

---

## 👨‍💻 Contributors
- **Mitansh Gor**  
- **Henry Ha**  
- **John Yun**

---

**License:** MIT  
**Version:** 1.0.0
