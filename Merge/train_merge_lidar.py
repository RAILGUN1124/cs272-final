"""
Training script for Merge-v0 environment with LidarObservation
"""
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import json
import os

class RewardCallback(BaseCallback):
    """
    Custom callback for tracking episode rewards during training
    """
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self._episode_timestamps = set()  # Track which episodes we've already processed
        
    def _on_step(self) -> bool:
        """
        Called at each step. Check for new episodes in the buffer.
        """
        # Process episodes in the buffer that we haven't seen yet
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                # Use timestamp as unique identifier for episodes
                ep_id = (ep_info['t'], ep_info['r'], ep_info['l'])
                
                if ep_id not in self._episode_timestamps:
                    self._episode_timestamps.add(ep_id)
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
                    self.episode_count += 1
                    
                    if self.verbose > 0 and self.episode_count % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        avg_reward = sum(recent_rewards) / len(recent_rewards)
                        print(f"Episode {self.episode_count}: Recent avg reward = {avg_reward:.2f}")
        
        return True
    
    def save_data(self, filename):
        """Save training data to JSON file"""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'num_episodes': len(self.episode_rewards)
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Training data saved to {filename}")

def create_merge_env_lidar():
    """Create Merge environment with LidarObservation"""
    env = gym.make('merge-v0', render_mode='rgb_array')
    
    # Configure environment for LidarObservation - optimized for Merge
    env.unwrapped.config.update({
        "observation": {
            "type": "LidarObservation",
            "cells": 16 * 4,  # 64 cells for detailed perception
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "policy_frequency": 2,
        "duration": 40,  # Episode length in steps
        "simulation_frequency": 15,
        "screen_width": 600,
        "screen_height": 150,
        "vehicles_count": 20,  # Reduced for merge scenario
        "collision_reward": -1,
        "right_lane_reward": 0.1,  # Reward for merging successfully
        "high_speed_reward": 0.4,  # Reward for maintaining speed
        "merging_speed_reward": -0.5,  # Small penalty if too slow
        "lane_change_reward": 0,
        "reward_speed_range": [20, 30],
        "offroad_terminal": True,
        "normalize_reward": True,
    })
    env.reset()  # Reset to apply configuration
    
    return env

def train_merge_lidar(timesteps=100000, model_name="merge_lidar"):
    """
    Train PPO agent on Merge-v0 with LidarObservation
    
    Args:
        timesteps: Number of training timesteps
        model_name: Name for saving the model
    """
    print("=" * 60)
    print("Training Merge-v0 with LidarObservation")
    print("=" * 60)
    
    # Create directories for saving results
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_dir = f"./logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment with Monitor wrapper for CSV logging
    def make_env():
        env = create_merge_env_lidar()
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create callback for tracking rewards
    callback = RewardCallback(verbose=1)
    
    # Create PPO model with optimized hyperparameters for Merge-v0
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3.5e-4,  # Slightly lower for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"./logs/{model_name}"
    )
    
    # Train the model
    print(f"\nStarting training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the trained model
    model_path = f"models/{model_name}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save training data
    callback.save_data(f"training_data/{model_name}_training.json")
    
    # Clean up
    env.close()
    
    return model, callback

if __name__ == "__main__":
    # Train the model
    model, callback = train_merge_lidar(timesteps=500000, model_name="merge_lidar")
    print("\nTraining completed successfully!")
