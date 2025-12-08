# Project Requirements - CS 272

## Objective

In this project, we will develop a toy autonomous driving vehicle using deep reinforcement learning (DRL). The goal is to design, train, and evaluate DRL agents capable of decision-making in simulated driving environments. We make use of our own custom environment which is the [highway_with_obstacles_env.py] which is added into the existing [highway-env package](https://highway-env.farama.org/) in the highway_env folder. The original framework provides a collection of benchmark environments for autonomous driving tasks based on gymnasium.

**Reference:** https://highway-env.farama.org/

---

## Project Structure

The project consists of two main tasks:

1. **DRL for Existing Environments** (Task 1)
2. **Design and Evaluation of a Custom Environment** (Task 2)

---

## Task 1: DRL for Existing Environments

### Environments
You will design and train DRL models to operate in the following three predefined environments:
- **Highway**
- **Merge**
- **Intersection**

### Requirements

#### Model Design
- Implement DRL agents capable of handling these environments.
- You are allowed to use and extend any open source DRL implementation, such as **stable-baselines3**.
- You can read recent papers and improve the performance by incorporating additional modules.
- **Important:** Make sure that you cite the original implementation and/or papers in your source code as comments and presentation slides.
- You may choose any of the existing action spaces provided by highway-env.

#### Observation Types
Each DRL model must be trained and evaluated under **two different observation types**:
- **LidarObservation**
- **GrayscaleObservation**

Reference: [Observation Types Documentation](https://highway-env.farama.org/)

#### Evaluation
For each environment and observation setup, plot:
1. **The learning curve** (Mean episodic training reward (return) vs. training episodes).
2. **The performance test result** of a trained model without exploration (A violin plot: Mean episodic training reward (return) for 500 episodes).

**Total plots:** 3 environments × 2 observation types × 2 plots = **12 plots total**

---

## Task 2: Design and Evaluation of a Custom Environment

### Requirements

#### Environment Design
- Create a new custom highway-env environment to introduce a new driving scenario.
- Custom env documentation: [HighwayEnv Custom Environments](https://highway-env.farama.org/)
- You may create the new environment by:
  - Adjusting the reward function to encourage a different driving behavior (e.g., safe lane changing, slower driving in a narrow street).
  - Altering the road configuration (e.g., roundabout, curved road, multi-lane junction, new obstacles).

#### Inter-Team Exchange
- You must share your custom environment with another team by **November 19** through github.
- Each team will receive a different custom environment from another group.

#### Evaluation
Train your DRL agent on both:
1. Your own custom environment.
2. The environment created by the other team.

For each environment, provide:
1. The learning curve.
2. The performance test result.

**Total plots:** 2 environments × 2 plots = **4 plots total**

---

## Submission Guidelines

### Source Code and Artifacts
Submit your:
- Source code
- Trained models
- TensorBoard data file
- Generated plots

**Important:** Create a public github repo and put the link on the front page of the presentation.

### Presentation (PDF)
Create a presentation based on the following items:

1. **Front page** (include team member names, github link, and your team ID)
2. **Task 1: DRL Design choices**
3. **Task 1: Results on Merge (LidarObservation)**
4. **Task 2a: Your custom env setup**
5. **Task 2a: DRL Design choices**
6. **Task 2a: Results on your custom env**
7. **Task 2b: The other team's custom env setup** (include the team ID you received the env from)
8. **Task 2b: DRL Design choices**
9. **Task 2b: Results on the other team's custom env**
10. **Appendix slides:**
    - 12 + 4 plots with clear labeling (see the following section about the experiment IDs.)

**Presentation Dates:** December 1 and 3 during the class. **(This is required to receive a passing grade.)**

**Note:** You do not need to present all ideas or results. Focus on the main ideas you think are unique and present main results. However, add all explanations and results in the appendix so I can read them later.

---

## Experiment Labeling

Whenever you save files (python code, trained model, tensorboard data file, and plots) or present results, please include the following ID in file names or slides.

### Experiment ID Table

| ID | Env | Obs | Result type |
|----|-----|-----|-------------|
| 1 | Highway | LidarObs | learning curve |
| 2 | Highway | LidarObs | performance test |
| 3 | Highway | GrayscaleObs | learning curve |
| 4 | Highway | GrayscaleObs | performance test |
| 5 | Merge | LidarObs | learning curve |
| 6 | Merge | LidarObs | performance test |
| 7 | Merge | GrayscaleObs | learning curve |
| 8 | Merge | GrayscaleObs | performance test |
| 9 | Intersection | LidarObs | learning curve |
| 10 | Intersection | LidarObs | performance test |
| 11 | Intersection | GrayscaleObs | learning curve |
| 12 | Intersection | GrayscaleObs | performance test |
| 13 | your custom | as defined by you | learning curve |
| 14 | your custom | as defined by you | performance test |
| 15 | other team's custom | as defined by the other team | learning curve |
| 16 | other team's custom | as defined by the other team | performance test |

### File Naming Convention
- Code: `experiment_{ID}_train.py`, `experiment_{ID}_eval.py`
- Models: `model_{ID}_*.pkl` or `model_{ID}_*.zip` (saved in `models/` directory)
- Plots: `plot_{ID}_learning_curve.png`, `plot_{ID}_performance_test.png` (saved in `plots/` directory)
- TensorBoard: Save logs in `logs/experiment_{ID}/`
- Results: Evaluation metrics and results (saved in `results/` directory)

**Note:** The project automatically creates `models/`, `logs/`, `plots/`, and `results/` directories when you run training scripts. These directories are included in `.gitignore` to avoid committing large files.

---

## RL Algorithm Setup

### Ready-to-Use Training Scripts

The project includes ready-to-use training scripts:

- **`scripts/train_ppo.py`** - Train PPO models
- **`scripts/train_sac.py`** - Train SAC models  
- **`scripts/train_dqn.py`** - Train DQN models
- **`scripts/train_all_models.py`** - Train all models for comparison

### Quick Training Examples

```bash
# Train PPO with LidarObservation (Experiment ID 1)
python scripts/train_ppo.py --env highway --obs lidar --timesteps 100000

# Train PPO with GrayscaleObservation (Experiment ID 3)
python scripts/train_ppo.py --env highway --obs grayscale --timesteps 100000

# Train on Merge environment with LidarObservation (Experiment ID 5)
python scripts/train_ppo.py --env merge --obs lidar --timesteps 100000
```

### Custom RL Setup

For custom implementations, use `scripts/train_rl.py` as a template. This file contains TODO comments indicating where to:
1. Initialize your RL algorithm (PPO, SAC, DQN, etc.)
2. Configure observation types (LidarObservation, GrayscaleObservation)
3. Set up training hyperparameters
4. Implement training and evaluation loops
5. Save models with proper experiment IDs

See [scripts/README.md](../scripts/README.md) for detailed documentation.
