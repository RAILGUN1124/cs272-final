"""
Training script for intersection-v0 environment with LidarObservation
Uses Transformer-based architecture for better interaction modeling
"""
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
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

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for LiDAR observations.
    Treats each LiDAR cell as a token to capture interactions between different readings.
    
    Reference: Attention Is All You Need (Vaswani et al., 2017)
    Applied to RL in: Stabilizing Transformers for Reinforcement Learning (Parisotto et al., 2019)
    https://arxiv.org/abs/1911.12250
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            observation_space: Environment observation space
            features_dim: Output feature dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate for regularization
        """
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # LiDAR observation is (n_cells, 2) where each cell has [distance, velocity]
        n_lidar_cells = observation_space.shape[0]
        lidar_feature_dim = observation_space.shape[1] if len(observation_space.shape) > 1 else 1
        
        # Embedding layer to project each LiDAR cell to higher dimension
        self.embedding_dim = 64
        self.input_projection = nn.Linear(lidar_feature_dim, self.embedding_dim)
        
        # Learnable positional encoding for each LiDAR cell position
        self.positional_encoding = nn.Parameter(torch.randn(1, n_lidar_cells, self.embedding_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(self.embedding_dim)
        
        # Output projection to features_dim
        self.output_projection = nn.Sequential(
            nn.Linear(self.embedding_dim * n_lidar_cells, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer feature extractor.
        
        Args:
            observations: LiDAR observations [batch_size, n_lidar_cells, lidar_features]
                         Each cell contains [distance, velocity] information
        
        Returns:
            features: Extracted features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # observations is already [batch_size, n_lidar_cells, lidar_features]
        # where lidar_features = 2 (distance and velocity for each cell)
        x = observations
        
        # Project each cell to embedding dimension
        x = self.input_projection(x)  # [batch_size, n_lidar_cells, embedding_dim]
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, n_lidar_cells, embedding_dim]
        
        # Layer normalization
        x = self.ln(x)
        
        # Flatten and project to output dimension
        x = x.reshape(batch_size, -1)  # [batch_size, n_lidar_cells * embedding_dim]
        features = self.output_projection(x)  # [batch_size, features_dim]
        
        return features

def create_intersection_env_lidar():
    """Create intersection environment with LidarObservation"""
    env = gym.make('intersection-v0', render_mode='rgb_array')
    
    # Configure environment for LidarObservation
    env.unwrapped.config.update({
        "observation": {
            "type": "LidarObservation",
            "cells": 16 * 4,  # 64 cells
        },
    })
    env.reset()  # Reset to apply configuration
    
    return env

def train_intersection_lidar(timesteps=1000000, model_name="intersection_lidar"):
    """
    Train PPO agent on intersection-v0 with LidarObservation
    
    Args:
        timesteps: Number of training timesteps (default 1M for optimal convergence)
        model_name: Name for saving the model
    """
    print("=" * 60)
    print("Training intersection-v0 with LidarObservation (PPO + Transformer)")
    print("=" * 60)
    
    # Create directories for saving results
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_dir = f"./logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create callback for tracking rewards
    callback = RewardCallback(verbose=1)
    
    # Create environment with Monitor wrapper for CSV logging
    def make_env():
        env = create_intersection_env_lidar()
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create custom policy with Transformer feature extractor
    print("\nInitializing PPO model with Transformer architecture...")
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            n_heads=4,  # 4 attention heads for capturing different interaction patterns
            n_layers=2,  # 2 transformer layers for hierarchical feature learning
            dropout=0.1  # Dropout for regularization
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64])  # Policy and value networks after feature extraction
    )
    
    # Create PPO model with optimized hyperparameters for intersection-v0
    model = PPO(
        "MlpPolicy",  # Base policy class (features will be extracted by Transformer)
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,  # Standard stable learning rate
        n_steps=512,  # Good balance for short episodes
        batch_size=128,  # Larger batch for stability
        n_epochs=10,  # Standard for PPO
        gamma=0.95,  # Lower gamma for shorter episode horizon
        gae_lambda=0.95,  # Standard GAE parameter
        clip_range=0.2,  # Standard PPO clip range
        clip_range_vf=None,  # No value function clipping
        ent_coef=0.01,  # Moderate entropy for exploration
        vf_coef=0.5,  # Standard value function coefficient
        max_grad_norm=0.5,  # Gradient clipping for stability
        use_sde=False,  # Don't use state-dependent exploration
        sde_sample_freq=-1,
        target_kl=None,  # No KL divergence constraint
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
    model, callback = train_intersection_lidar(timesteps=500000, model_name="intersection_lidar")
    print("\nTraining completed successfully!")
