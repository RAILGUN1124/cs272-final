"""
Training script for intersection-v0 environment with GrayscaleObservation
Uses Vision Transformer (ViT) architecture for better spatial relationship modeling
"""
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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

class VisionTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Vision Transformer (ViT) feature extractor for grayscale image observations.
    Divides image into patches and treats each patch as a token.
    
    Reference: An Image is Worth 16x16 Words (Dosovskiy et al., 2020)
    https://arxiv.org/abs/2010.11929
    Applied to RL in: Stabilizing Transformers for Reinforcement Learning (Parisotto et al., 2019)
    https://arxiv.org/abs/1911.12250
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 patch_size: int = 8, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            observation_space: Environment observation space (C, H, W)
            features_dim: Output feature dimension
            patch_size: Size of image patches (e.g., 8x8)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate for regularization
        """
        super(VisionTransformerFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Observation shape: (channels, height, width)
        n_channels = observation_space.shape[0]
        height = observation_space.shape[1]
        width = observation_space.shape[2]
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.n_patches_h = height // patch_size
        self.n_patches_w = width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # Embedding dimension for transformer
        self.embedding_dim = 128
        
        # Patch embedding: Conv2d to extract patches and embed them
        self.patch_embedding = nn.Conv2d(
            in_channels=n_channels,
            out_channels=self.embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable positional encoding for each patch
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_patches, self.embedding_dim))
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            dim_feedforward=512,
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
            nn.Linear(self.embedding_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer feature extractor.
        
        Args:
            observations: Image observations [batch_size, channels, height, width]
        
        Returns:
            features: Extracted features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # Extract patches and embed them
        # [batch_size, channels, height, width] -> [batch_size, embedding_dim, n_patches_h, n_patches_w]
        x = self.patch_embedding(observations)
        
        # Reshape to sequence of patches
        # [batch_size, embedding_dim, n_patches_h, n_patches_w] -> [batch_size, n_patches, embedding_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, n_patches + 1, embedding_dim]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Layer normalization
        x = self.ln(x)
        
        # Use CLS token output as global representation
        cls_output = x[:, 0]  # [batch_size, embedding_dim]
        
        # Project to output dimension
        features = self.output_projection(cls_output)  # [batch_size, features_dim]
        
        return features

def create_intersection_env_grayscale():
    """Create intersection environment with GrayscaleObservation"""
    env = gym.make('intersection-v0', render_mode='rgb_array')
    
    # Configure environment for GrayscaleObservation
    env.unwrapped.config.update({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # Weights for RGB to grayscale conversion
            "scaling": 1.75,
        },
    })
    env.reset()  # Reset to apply configuration
    
    return env

def train_intersection_grayscale(timesteps=1000000, model_name="intersection_grayscale"):
    """
    Train PPO agent on intersection-v0 with GrayscaleObservation
    
    Args:
        timesteps: Number of training timesteps (default 1M for optimal convergence)
        model_name: Name for saving the model
    """
    print("=" * 60)
    print("Training intersection-v0 with GrayscaleObservation (PPO + Vision Transformer)")
    print("=" * 60)
    
    # Create directories for saving results
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_dir = f"./logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment with Monitor wrapper for CSV logging
    def make_env():
        env = create_intersection_env_grayscale()
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create callback for tracking rewards
    callback = RewardCallback(verbose=1)
    
    # Create custom policy with Vision Transformer feature extractor
    print("\nInitializing PPO model with Vision Transformer architecture...")
    policy_kwargs = dict(
        features_extractor_class=VisionTransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            patch_size=8,  # 8x8 patches (128x64 image -> 16x8 = 128 patches)
            n_heads=4,  # 4 attention heads for capturing different spatial patterns
            n_layers=2,  # 2 transformer layers for hierarchical feature learning
            dropout=0.1  # Dropout for regularization
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128])  # Policy and value networks after feature extraction
    )
    
    # Create PPO model with optimized hyperparameters
    model = PPO(
        "CnnPolicy",  # Base policy class (features will be extracted by Vision Transformer)
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,  # Lower LR for stability with transformers
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
    model, callback = train_intersection_grayscale(timesteps=500000, model_name="intersection_grayscale")
    print("\nTraining completed successfully!")
