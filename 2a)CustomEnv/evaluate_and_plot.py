"""
Evaluation and plotting script for NarrowLaneSafeChange-v0 DRL models
Generates learning curves and violin plots for model performance
"""
import gymnasium as gym
from stable_baselines3 import PPO, DQN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm

# Import custom environment
import __init__

def create_env():
    """Create NarrowLaneSafeChange environment"""
    env = gym.make('NarrowLaneSafeChange-v0', render_mode='rgb_array')
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

def load_monitor_csv(log_path):
    """Load episode rewards from monitor.csv"""
    import pandas as pd
    monitor_file = os.path.join(log_path, "monitor.csv")
    if not os.path.exists(monitor_file):
        return None
    try:
        df = pd.read_csv(monitor_file, skiprows=1)
        if 'r' in df.columns:
            return df['r'].tolist()
    except Exception as e:
        print(f"Warning: Could not read monitor.csv: {e}")
    return None

def main():
    """Main evaluation and plotting function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and plot NarrowLaneSafeChange-v0 models")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["dqn", "ppo"], help="Algorithm type")
    parser.add_argument("--training-data", type=str, help="Path to training data JSON (optional)")
    parser.add_argument("--log-path", type=str, help="Path to training logs directory with monitor.csv (auto-detects if not provided)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NarrowLaneSafeChange-v0 Model Evaluation and Plotting")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("evaluation_data", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    
    model_name = os.path.basename(args.model_path)
    
    # Auto-detect log path if not provided
    if not args.log_path:
        args.log_path = f"./logs/{args.algorithm}_narrow_lane"
    
    # Try to load training data for learning curve
    training_rewards = None
    
    # First, try provided training data JSON
    if args.training_data and os.path.exists(args.training_data):
        print(f"\n1. Loading training data from {args.training_data}...")
        plot_learning_curve(
            args.training_data,
            f"{args.output_dir}/{model_name}_learning_curve.png",
            f"NarrowLaneSafeChange-v0 Learning Curve ({args.algorithm.upper()})"
        )
    # Second, try to load from monitor.csv
    elif os.path.exists(args.log_path):
        print(f"\n1. Looking for training logs in {args.log_path}...")
        training_rewards = load_monitor_csv(args.log_path)
        if training_rewards:
            print(f"   Found {len(training_rewards)} training episodes in monitor.csv")
            # Save to JSON for future use
            training_data = {
                'episode_rewards': training_rewards,
                'num_episodes': len(training_rewards),
                'algorithm': args.algorithm
            }
            json_path = f"training_data/{args.algorithm}_training.json"
            with open(json_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            print(f"   Saved training data to {json_path}")
            
            # Plot learning curve
            plot_learning_curve(
                json_path,
                f"{args.output_dir}/{model_name}_learning_curve.png",
                f"NarrowLaneSafeChange-v0 Learning Curve ({args.algorithm.upper()})"
            )
        else:
            print(f"   No monitor.csv found in {args.log_path}")
    else:
        print(f"\n1. No training data available (tried: {args.training_data}, {args.log_path})")
    
    # Load and evaluate model
    step_num = "\n2." if training_rewards or (args.training_data and os.path.exists(args.training_data)) else "\n1."
    print(f"{step_num} Evaluating trained model...")
    if args.algorithm.lower() == "dqn":
        model = DQN.load(args.model_path)
    elif args.algorithm.lower() == "ppo":
        model = PPO.load(args.model_path)
    
    eval_rewards = evaluate_model(model, create_env, n_episodes=args.episodes)
    
    # Save evaluation data
    eval_file = f"evaluation_data/{model_name}_eval.json"
    with open(eval_file, 'w') as f:
        json.dump({
            'rewards': eval_rewards,
            'mean': float(np.mean(eval_rewards)),
            'std': float(np.std(eval_rewards)),
            'median': float(np.median(eval_rewards)),
            'min': float(np.min(eval_rewards)),
            'max': float(np.max(eval_rewards)),
            'algorithm': args.algorithm
        }, f, indent=2)
    print(f"Evaluation data saved to {eval_file}")
    
    # Plot violin plot
    step_num = "\n3." if training_rewards or (args.training_data and os.path.exists(args.training_data)) else "\n2."
    print(f"{step_num} Generating violin plot...")
    plot_violin_performance(
        eval_rewards,
        f"{args.output_dir}/{model_name}_violin.png",
        f"NarrowLaneSafeChange-v0 Performance - {args.episodes} Episodes ({args.algorithm.upper()})"
    )
    
    print(f"\n{args.algorithm.upper()} Model Results:")
    print(f"  Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Median Reward: {np.median(eval_rewards):.2f}")
    print(f"  Min/Max: {np.min(eval_rewards):.2f} / {np.max(eval_rewards):.2f}")
    
    print("\n" + "=" * 60)
    print("Evaluation and plotting completed!")
    print("=" * 60)
    print(f"\nGenerated plots saved in '{args.output_dir}/' directory")

if __name__ == "__main__":
    main()
