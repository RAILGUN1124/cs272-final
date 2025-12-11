# NarrowLaneSafeChange-v0 Environment

A custom Highway-Env environment focused on safe lane-changing behavior in a narrow two-lane highway with traffic vehicles.

## Overview

This environment models a constrained road scenario where the agent must navigate a two-lane highway with traffic vehicles that stay in their lanes. The focus is on:
- **Collision avoidance** with stationary-lane traffic
- **Lane change execution** when needed
- **Safe navigation** through traffic

## Environment Details

### Road Structure
- **1 lanes** in the same direction (`a → b`), another "ghost" lane to allow the controlled vehicle to takeover other vehicles
- **1 oncoming lane** for traffic (`b → a`)
- **10 traffic vehicles** spawned with random speeds (5-20 m/s)
- Traffic vehicles **do not change lanes** (disabled via lane change parameters)
- **1000m long** straight highway

### Observation Space
**Type**: Kinematics (normalized)
- Observes **5 nearest vehicles** (including ego)
- **Features per vehicle**: `[presence, x, y, vx, vy]`
- **Relative coordinates** (not absolute)
- **Can see behind** the ego vehicle
- All values **normalized**

### Action Space
**Type**: DiscreteMetaAction (5 actions)
- `0`: **LANE_LEFT** - Change to left lane
- `1`: **IDLE** - Maintain speed and lane
- `2`: **LANE_RIGHT** - Change to right lane
- `3`: **FASTER** - Accelerate
- `4`: **SLOWER** - Decelerate

### Reward Function

| Component | Weight | Behavior | Purpose |
|-----------|--------|----------|---------|
| Collision | 0 | Applied when crashed | Penalty for collisions (episode ends) |
| Survival | 0.1 | +1.0 per timestep alive | Encourages staying collision-free |
| Lane change | 0.2 | +1.0 when lane changes | Rewards lane change actions |
| Unsafe lane change | 0 | 0 (disabled) | Was for detecting risky merges |

**Normalization**: Rewards are normalized to `[0, 1]` range and clipped to `[-1, 1]`.

**Note**: Despite config values, actual penalties like `unsafe_lane_change_penalty` and `speed_penalty_factor` are set to 0, meaning they don't affect the reward in practice.

### Episode Termination
- **Terminated**: Collision with another vehicle
- **Truncated**: After 40 seconds (80 steps at policy frequency of 2 Hz)

### Key Parameters
- **Duration**: 40 seconds
- **Policy frequency**: 2 Hz (agent acts twice per second)
- **Simulation frequency**: 10 Hz
- **Max speed**: 25 m/s (~90 km/h)
- **Min speed**: 5 m/s (~18 km/h)
- **Traffic count**: 10 vehicles
- **Lane width**: 3.5 meters

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```
## Citation

Built on top of:
- [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

## License

MIT License - feel free to modify and extend this environment for your research or projects.
