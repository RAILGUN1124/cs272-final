"""
Evaluation and plotting script for Highway-v0 DRL models
Generates learning curves and violin plots for model performance
"""
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm

def create_highway_env_lidar():
    """Create Highway environment with LidarObservation"""
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.unwrapped.config.update({
        "observation": {
            "type": "LidarObservation",
            "cells": 16 * 4,
        },
        # "policy_frequency": 2,
        # "duration": 40,
        # "vehicles_count": 50,
        # "collision_reward": -1,
        # "reward_speed_range": [20, 30],
        # "simulation_frequency": 15,
        # "lanes_count": 4,
        # "initial_lane_id": None,
        # "offroad_terminal": True,
    })
    env.reset()  # Reset to apply configuration
    return env

def create_highway_env_grayscale():
    """Create Highway environment with GrayscaleObservation"""
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.unwrapped.config.update({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
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

def evaluate_model(model, env_creator, n_episodes=100, deterministic=True):
    """
    Evaluate a trained model for n_episodes
    
    Args:
        model: Trained model
        env_creator: Function to create environment
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions (no exploration)
        
    Returns:
        List of episode rewards
    """
    env = env_creator()
    episode_rewards = []
    
    print(f"Evaluating model for {n_episodes} episodes (deterministic={deterministic})...")
    
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    env.close()
    return episode_rewards

def plot_learning_curve(training_data_path, save_path, title):
    """
    Plot learning curve from training data
    
    Args:
        training_data_path: Path to training data JSON file
        save_path: Path to save the plot
        title: Title for the plot
    """
    # Load training data
    with open(training_data_path, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data['episode_rewards']
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # Calculate moving average for smoother curve
    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_episodes = episodes[window_size-1:]
    else:
        moving_avg = episode_rewards
        moving_avg_episodes = episodes
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    plt.plot(moving_avg_episodes, moving_avg, label=f'Moving Average (window={window_size})', 
             color='red', linewidth=2)
    
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Mean Episodic Reward (Return)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curve saved to {save_path}")
    plt.close()

def plot_violin_performance(eval_rewards, save_path, title):
    """
    Plot violin plot for evaluation performance
    
    Args:
        eval_rewards: List of episode rewards from evaluation
        save_path: Path to save the plot
        title: Title for the plot
    """
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Create violin plot
    parts = plt.violinplot([eval_rewards], positions=[0], showmeans=True, 
                           showmedians=True, widths=0.7)
    
    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    
    # Add statistics text
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    median_reward = np.median(eval_rewards)
    min_reward = np.min(eval_rewards)
    max_reward = np.max(eval_rewards)
    
    stats_text = f'Mean: {mean_reward:.2f}\nStd: {std_reward:.2f}\nMedian: {median_reward:.2f}\nMin: {min_reward:.2f}\nMax: {max_reward:.2f}'
    plt.text(0.5, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.ylabel('Mean Episodic Reward (Return)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks([0], ['Trained Model (No Exploration)'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to {save_path}")
    plt.close()

def main():
    """Main evaluation and plotting function"""
    print("=" * 60)
    print("Highway-v0 Model Evaluation and Plotting")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    os.makedirs("evaluation_data", exist_ok=True)
    
    # Process LidarObservation model
    print("\n" + "=" * 60)
    print("Processing LidarObservation Model")
    print("=" * 60)
    
    if os.path.exists("models/highway_lidar.zip") and os.path.exists("training_data/highway_lidar_training.json"):
        # Plot learning curve
        print("\n1. Generating learning curve...")
        plot_learning_curve(
            "training_data/highway_lidar_training.json",
            "plots/highway_lidar_learning_curve.png",
            "Highway-v0 Learning Curve (LidarObservation)"
        )
        
        # Evaluate model
        print("\n2. Evaluating trained model...")
        model_lidar = PPO.load("models/highway_lidar")
        eval_rewards_lidar = evaluate_model(model_lidar, create_highway_env_lidar, n_episodes=100)
        
        # Save evaluation data
        with open("evaluation_data/highway_lidar_eval.json", 'w') as f:
            json.dump({
                'rewards': eval_rewards_lidar,
                'mean': float(np.mean(eval_rewards_lidar)),
                'std': float(np.std(eval_rewards_lidar)),
                'median': float(np.median(eval_rewards_lidar)),
                'min': float(np.min(eval_rewards_lidar)),
                'max': float(np.max(eval_rewards_lidar))
            }, f)
        
        # Plot violin plot
        print("\n3. Generating violin plot...")
        plot_violin_performance(
            eval_rewards_lidar,
            "plots/highway_lidar_violin.png",
            "Highway-v0 Performance Test - 100 Episodes (LidarObservation)"
        )
        
        print(f"\nLidarObservation Results:")
        print(f"  Mean Reward: {np.mean(eval_rewards_lidar):.2f} ± {np.std(eval_rewards_lidar):.2f}")
        print(f"  Median Reward: {np.median(eval_rewards_lidar):.2f}")
        print(f"  Min/Max: {np.min(eval_rewards_lidar):.2f} / {np.max(eval_rewards_lidar):.2f}")
    else:
        print("⚠ LidarObservation model or training data not found. Skipping...")
    
    # Process GrayscaleObservation model
    print("\n" + "=" * 60)
    print("Processing GrayscaleObservation Model")
    print("=" * 60)
    
    if os.path.exists("models/highway_grayscale.zip") and os.path.exists("training_data/highway_grayscale_training.json"):
        # Plot learning curve
        print("\n1. Generating learning curve...")
        plot_learning_curve(
            "training_data/highway_grayscale_training.json",
            "plots/highway_grayscale_learning_curve.png",
            "Highway-v0 Learning Curve (GrayscaleObservation)"
        )
        
        # Evaluate model
        print("\n2. Evaluating trained model...")
        model_grayscale = PPO.load("models/highway_grayscale")
        eval_rewards_grayscale = evaluate_model(model_grayscale, create_highway_env_grayscale, n_episodes=100)
        
        # Save evaluation data
        with open("evaluation_data/highway_grayscale_eval.json", 'w') as f:
            json.dump({
                'rewards': eval_rewards_grayscale,
                'mean': float(np.mean(eval_rewards_grayscale)),
                'std': float(np.std(eval_rewards_grayscale)),
                'median': float(np.median(eval_rewards_grayscale)),
                'min': float(np.min(eval_rewards_grayscale)),
                'max': float(np.max(eval_rewards_grayscale))
            }, f)
        
        # Plot violin plot
        print("\n3. Generating violin plot...")
        plot_violin_performance(
            eval_rewards_grayscale,
            "plots/highway_grayscale_violin.png",
            "Highway-v0 Performance Test - 100 Episodes (GrayscaleObservation)"
        )
        
        print(f"\nGrayscaleObservation Results:")
        print(f"  Mean Reward: {np.mean(eval_rewards_grayscale):.2f} ± {np.std(eval_rewards_grayscale):.2f}")
        print(f"  Median Reward: {np.median(eval_rewards_grayscale):.2f}")
        print(f"  Min/Max: {np.min(eval_rewards_grayscale):.2f} / {np.max(eval_rewards_grayscale):.2f}")
    else:
        print("⚠ GrayscaleObservation model or training data not found. Skipping...")
    
    print("\n" + "=" * 60)
    print("Evaluation and plotting completed!")
    print("=" * 60)
    print("\nGenerated plots are saved in the 'plots/' directory:")
    print("  - highway_lidar_learning_curve.png")
    print("  - highway_lidar_violin.png")
    print("  - highway_grayscale_learning_curve.png")
    print("  - highway_grayscale_violin.png")

if __name__ == "__main__":
    main()
