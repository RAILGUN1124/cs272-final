# Highway Environment Deep Reinforcement Learning

This project trains and evaluates PPO agents on the Highway-v0 environment using two different observation types: **Grayscale** and **LiDAR**.

## Prerequisites

```bash
pip install gymnasium highway-env stable-baselines3 matplotlib seaborn numpy tqdm
```

## How to Run

### Training

Train a PPO agent with grayscale observation:
```bash
python train_highway_grayscale.py
```

Train a PPO agent with LiDAR observation:
```bash
python train_highway_lidar.py
```

Both training scripts will:
- Train for 100,000 timesteps
- Save models to `models/` directory
- Log training data to `training_data/` directory
- Generate TensorBoard logs in `logs/` directory

### Evaluation

Evaluate trained models and generate plots:
```bash
python evaluate_and_plot.py
```

This script will:
- Load trained models from `models/` directory
- Run evaluation episodes
- Save evaluation results to `evaluation_data/`
- Generate learning curves and violin plots in `plots/` directory

## File Structure

```
.
├── train_highway_grayscale.py   # Training script for grayscale observation
├── train_highway_lidar.py       # Training script for LiDAR observation
├── evaluate_and_plot.py         # Evaluation and visualization script
├── models/                      # Saved PPO models (.zip files)
├── logs/                        # TensorBoard event logs
│   ├── highway_grayscale/       # Grayscale training logs
│   └── highway_lidar/           # LiDAR training logs
├── training_data/               # Training metrics (JSON)
│   ├── highway_grayscale_training.json
│   └── highway_lidar_training.json
├── evaluation_data/             # Evaluation metrics (JSON)
│   ├── highway_grayscale_eval.json
│   └── highway_lidar_eval.json
└── plots/                       # Generated visualization plots
```

## Observation Types

- **Grayscale**: Uses visual observation (128x64 pixels, 4-frame stack)
- **LiDAR**: Uses 64-cell LiDAR sensor readings

## Model Architecture

Both models use PPO (Proximal Policy Optimization) with:
- Multi-layer perceptron (MLP) policy
- Custom reward callbacks for tracking training progress
- Episode monitoring and logging
