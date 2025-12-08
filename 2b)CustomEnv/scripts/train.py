"""
Advanced Training Script for Highway Environment with Construction Zones
Supports multiple RL algorithms: DQN, PPO, SAC, A2C
Features: checkpoint saving, TensorBoard logging, hyperparameter tuning, curriculum learning
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import highway_env  # This registers all custom environments


class ProgressCallback(BaseCallback):
    """Custom callback for tracking training progress with detailed metrics"""
    
    def __init__(self, save_path=None, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_count = 0
        self.success_count = 0
        self.episode_count = 0
        self.save_path = save_path
        
    def _on_step(self) -> bool:
        # Track episode-level metrics
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            
            # Get episode reward from info or rollout buffer
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
            
            # Track collisions
            if info.get("crashed", False):
                self.collision_count += 1
            
            # Track success (completed without collision)
            if not info.get("crashed", False):
                self.success_count += 1
            
            self.episode_count += 1
            
            # Log metrics every 100 episodes
            if self.episode_count % 100 == 0:
                collision_rate = self.collision_count / self.episode_count
                success_rate = self.success_count / self.episode_count
                
                self.logger.record("metrics/collision_rate", collision_rate)
                self.logger.record("metrics/success_rate", success_rate)
                self.logger.record("metrics/episodes", self.episode_count)
                
                if self.verbose > 0:
                    print(f"Episodes: {self.episode_count}, "
                          f"Collision Rate: {collision_rate:.2%}, "
                          f"Success Rate: {success_rate:.2%}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Generate learning curve plot at end of training"""
        if len(self.episode_rewards) > 0 and self.save_path:
            self._plot_learning_curve()
    
    def _plot_learning_curve(self):
        """Generate and save learning curve plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        
        # Plot raw rewards
        ax.plot(episodes, self.episode_rewards, alpha=0.3, linewidth=0.5, label='Episode Reward')
        
        # Plot moving average (window=50)
        if len(self.episode_rewards) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, linewidth=2, color='red', label=f'Moving Average (window={window})')
        
        ax.set_xlabel('Training Episodes', fontsize=12)
        ax.set_ylabel('Mean Episodic Return (Reward)', fontsize=12)
        ax.set_title('Learning Curve: Training Progress', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(self.save_path) / 'learning_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nLearning curve saved to: {plot_path}")


def create_env(env_id: str = "highway-with-obstacles-v0", 
               config: Optional[Dict[str, Any]] = None,
               render_mode: Optional[str] = None) -> gym.Env:
    """Create and configure the environment"""
    
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        
        # Apply custom configuration
        if config:
            env.unwrapped.config.update(config)
        
        # Wrap with Monitor for episode statistics
        env = Monitor(env)
        return env
    
    return _init


def get_default_env_config(difficulty: str = "medium") -> Dict[str, Any]:
    """Get default environment configuration based on difficulty"""
    
    base_config = {
        "duration": 40,  # [s]
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
    }
    
    if difficulty == "easy":
        config = {
            **base_config,
            "obstacles_count": 5,
            "obstacle_spacing": 30,
            "vehicles_count": 10,
            "construction_zones_count": 1,
            "construction_zone_length": 100,
            "construction_zone_side": "left",
            "construction_zone_lanes": 1,
            "construction_cone_spacing": 8,
        }
    elif difficulty == "medium":
        config = {
            **base_config,
            "obstacles_count": 10,
            "obstacle_spacing": 20,
            "vehicles_count": 20,
            "construction_zones_count": 2,
            "construction_zone_length": 150,
            "construction_zone_side": "random",
            "construction_zone_lanes": 2,
            "construction_cone_spacing": 5,
        }
    else:  # hard
        config = {
            **base_config,
            "obstacles_count": 20,
            "obstacle_spacing": 15,
            "vehicles_count": 30,
            "construction_zones_count": 3,
            "construction_zone_length": 200,
            "construction_zone_side": "random",
            "construction_zone_lanes": 2,
            "construction_cone_spacing": 4,
        }
    
    # Add reward configuration
    config["reward"] = {
        "collision_penalty": -1.0,
        "closed_lane_penalty": -1.0,
        "speed_compliance": {"within_limit": 0.05},
        "speed_violation": {"beyond_limit": -0.05}
    }
    
    return config


def get_algorithm_hyperparams(algorithm: str, custom_params: Optional[Dict] = None) -> Dict[str, Any]:
    """Get default hyperparameters for each algorithm"""
    
    hyperparams = {
        "dqn": {
            "policy": "MlpPolicy",
            "learning_rate": 5e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.95,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 500,
            "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": dict(net_arch=[256, 256]),
        },
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(net_arch=[256, 256]),
        },
        "sac": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "policy_kwargs": dict(net_arch=[256, 256]),
        },
        "a2c": {
            "policy": "MlpPolicy",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(net_arch=[256, 256]),
        }
    }
    
    params = hyperparams.get(algorithm.lower(), hyperparams["ppo"])
    
    # Update with custom parameters
    if custom_params:
        params.update(custom_params)
    
    return params


def train(
    algorithm: str = "ppo",
    total_timesteps: int = 100000,
    env_config: Optional[Dict[str, Any]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    n_envs: int = 4,
    save_dir: str = "models",
    log_dir: str = "logs",
    eval_freq: int = 10000,
    save_freq: int = 10000,
    difficulty: str = "medium",
    seed: int = 0,
    verbose: int = 1,
) -> None:
    """
    Train an RL agent on the highway environment
    
    Args:
        algorithm: RL algorithm to use (dqn, ppo, sac, a2c)
        total_timesteps: Total training timesteps
        env_config: Custom environment configuration
        hyperparams: Custom hyperparameters for the algorithm
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: Directory for TensorBoard logs
        eval_freq: Frequency of evaluation
        save_freq: Frequency of model checkpoints
        difficulty: Environment difficulty (easy, medium, hard)
        seed: Random seed
        verbose: Verbosity level
    """
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{algorithm}_{difficulty}_{timestamp}"
    save_path = Path(save_dir) / exp_name
    log_path = Path(log_dir) / exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Training Configuration")
    print("=" * 80)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Difficulty: {difficulty}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Save directory: {save_path}")
    print(f"Log directory: {log_path}")
    print(f"Random seed: {seed}")
    print("=" * 80)
    
    # Get environment configuration
    if env_config is None:
        env_config = get_default_env_config(difficulty)
    
    # Save configuration
    config_path = save_path / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "algorithm": algorithm,
            "difficulty": difficulty,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "seed": seed,
            "env_config": env_config,
            "hyperparams": hyperparams or {},
        }, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Create vectorized environment
    print("\nCreating environment...")
    if n_envs > 1:
        vec_env = make_vec_env(
            create_env("highway-with-obstacles-v0", env_config),
            n_envs=n_envs,
            seed=seed,
            vec_env_cls=SubprocVecEnv if n_envs > 4 else DummyVecEnv
        )
    else:
        vec_env = DummyVecEnv([create_env("highway-with-obstacles-v0", env_config)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_env("highway-with-obstacles-v0", env_config)])
    
    # Get algorithm hyperparameters
    params = get_algorithm_hyperparams(algorithm, hyperparams)
    
    print(f"\nInitializing {algorithm.upper()} model...")
    print(f"Hyperparameters: {json.dumps(params, indent=2)}")
    
    # Create model
    algorithm_class = {
        "dqn": DQN,
        "ppo": PPO,
        "sac": SAC,
        "a2c": A2C,
    }[algorithm.lower()]
    
    model = algorithm_class(
        env=vec_env,
        verbose=verbose,
        tensorboard_log=str(log_path),
        seed=seed,
        **params
    )
    
    # Configure logger
    new_logger = configure(str(log_path), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix=algorithm,
        save_replay_buffer=True if algorithm.lower() in ["dqn", "sac"] else False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(log_path),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Progress callback
    progress_callback = ProgressCallback(save_path=str(save_path), verbose=verbose)
    callbacks.append(progress_callback)
    
    callback = CallbackList(callbacks)
    
    # Train the model
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = save_path / "final_model"
        model.save(str(final_model_path))
        print(f"\n{'=' * 80}")
        print(f"Training completed!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Best model saved to: {save_path / 'best_model'}")
        print(f"Checkpoints saved to: {save_path / 'checkpoints'}")
        print(f"TensorBoard logs: {log_path}")
        print(f"{'=' * 80}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_model_path = save_path / "interrupted_model"
        model.save(str(interrupted_model_path))
        print(f"Model saved to: {interrupted_model_path}")
    
    finally:
        vec_env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on Highway Environment with Construction Zones"
    )
    
    # Algorithm settings
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="ppo",
        choices=["dqn", "ppo", "sac", "a2c"],
        help="RL algorithm to use"
    )
    
    # Training settings
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--n-envs", "-n",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Environment difficulty level"
    )
    
    # Paths
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs"
    )
    
    # Callbacks
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency (in timesteps)"
    )
    
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Model checkpoint frequency (in timesteps)"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config JSON file"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    env_config = None
    if args.config:
        with open(args.config, "r") as f:
            env_config = json.load(f)
    
    # Run training
    train(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        env_config=env_config,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        difficulty=args.difficulty,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
