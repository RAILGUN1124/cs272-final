"""
Training Script for NarrowLaneSafeChange-v0 Environment

Train a DQN agent using Stable Baselines3 to navigate the narrow lane environment.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys

# Import custom environment
import __init__  # This registers the environment


def make_env():
    """Create and configure the environment."""
    env = gym.make('NarrowLaneSafeChange-v0', render_mode='rgb_array')
    return env


def train_dqn(
    total_timesteps: int = 100_000,
    learning_rate: float = 5e-4,
    buffer_size: int = 50_000,
    learning_starts: int = 1000,
    batch_size: int = 128,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    save_path: str = "./models/dqn_narrow_lane",
    log_path: str = "./logs/dqn_narrow_lane"
):
    """
    Train a DQN agent on the NarrowLaneSafeChange environment.
    
    Args:
        total_timesteps: Total number of training timesteps
        learning_rate: Learning rate for the optimizer
        buffer_size: Size of the replay buffer
        learning_starts: Number of steps before learning starts
        batch_size: Minibatch size for training
        gamma: Discount factor
        target_update_interval: Update target network every N steps
        exploration_fraction: Fraction of training for exploration
        exploration_final_eps: Final epsilon for exploration
        save_path: Path to save model checkpoints
        log_path: Path to save training logs
    """
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs("./models/best", exist_ok=True)
    
    # Create training environment
    print("Creating training environment...")
    env = DummyVecEnv([make_env])
    env = VecMonitor(env, log_path)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    
    # Create DQN model
    print("Initializing DQN model...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=log_path,
        device="auto"
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path,
        name_prefix="dqn_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best",
        log_path=log_path,
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=100,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, "dqn_final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate final model
    print("\nEvaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


def train_ppo(
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    save_path: str = "./models/ppo_narrow_lane",
    log_path: str = "./logs/ppo_narrow_lane"
):
    """
    Train a PPO agent on the NarrowLaneSafeChange environment.
    
    PPO often works well for continuous control and complex reward structures.
    """
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs("./models/best_ppo", exist_ok=True)
    
    # Create training environment
    print("Creating training environment...")
    env = DummyVecEnv([make_env])
    env = VecMonitor(env, log_path)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log=log_path,
        device="auto"
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path,
        name_prefix="ppo_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_ppo",
        log_path=log_path,
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, "ppo_final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate final model
    print("\nEvaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent on NarrowLaneSafeChange-v0")
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="dqn", 
        choices=["dqn", "ppo"],
        help="RL algorithm to use (dqn or ppo)"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=100_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=None,
        help="Learning rate (defaults: 5e-4 for DQN, 3e-4 for PPO)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Training {args.algorithm.upper()} on NarrowLaneSafeChange-v0")
    print("=" * 60)
    
    if args.algorithm == "dqn":
        lr = args.learning_rate if args.learning_rate else 5e-4
        model = train_dqn(
            total_timesteps=args.timesteps,
            learning_rate=lr
        )
    elif args.algorithm == "ppo":
        lr = args.learning_rate if args.learning_rate else 3e-4
        model = train_ppo(
            total_timesteps=args.timesteps,
            learning_rate=lr
        )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
