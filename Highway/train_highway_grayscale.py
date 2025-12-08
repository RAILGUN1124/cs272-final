"""
Training script for Highway-v0 environment with GrayscaleObservation
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

def create_highway_env_grayscale():
    """Create Highway environment with GrayscaleObservation"""
    env = gym.make('highway-v0', render_mode='rgb_array')
    
    # Configure environment for GrayscaleObservation
    env.unwrapped.config.update({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # Weights for RGB to grayscale conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 40,
        "vehicles_count": 50,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,
        "lanes_count": 4,
        "initial_lane_id": None,
        "offroad_terminal": True,
    })
    env.reset()  # Reset to apply configuration
    
    return env

def train_highway_grayscale(timesteps=100000, model_name="highway_grayscale"):
    """
    Train PPO agent on Highway-v0 with GrayscaleObservation
    
    Args:
        timesteps: Number of training timesteps
        model_name: Name for saving the model
    """
    print("=" * 60)
    print("Training Highway-v0 with GrayscaleObservation")
    print("=" * 60)
    
    # Create directories for saving results
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_dir = f"./logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment with Monitor wrapper for CSV logging
    def make_env():
        env = create_highway_env_grayscale()
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create callback for tracking rewards
    callback = RewardCallback(verbose=1)
    
    # Create PPO model with CNN policy for image observations
    print("\nInitializing PPO model with CnnPolicy...")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"./logs/{model_name}",
        device="cuda"  # Use GPU if available, falls back to CPU
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
    model, callback = train_highway_grayscale(timesteps=100000, model_name="highway_grayscale")
    print("\nTraining completed successfully!")
