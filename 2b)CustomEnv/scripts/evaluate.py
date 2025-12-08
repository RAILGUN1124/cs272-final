"""
Comprehensive Evaluation Script for Highway Environment
Features: detailed metrics, video recording, statistical analysis, visualization
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.evaluation import evaluate_policy

import highway_env  # This registers all custom environments


class EvaluationMetrics:
    """Track and compute evaluation metrics"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.collisions = []
        self.speeds = []
        self.lane_changes = []
        self.construction_zone_violations = []
        self.success_episodes = []
        self.episode_info = []
        
    def add_episode(self, reward: float, length: int, info: Dict[str, Any]):
        """Add episode data"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.collisions.append(1 if info.get("crashed", False) else 0)
        self.success_episodes.append(1 if not info.get("crashed", False) else 0)
        self.episode_info.append(info)
    
    def add_step_data(self, speed: float, lane_change: bool = False, 
                     zone_violation: bool = False):
        """Add step-level data"""
        self.speeds.append(speed)
        if lane_change:
            self.lane_changes.append(1)
        if zone_violation:
            self.construction_zone_violations.append(1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics"""
        stats = {
            # Episode-level metrics
            "num_episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "median_reward": np.median(self.episode_rewards),
            
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            
            # Safety metrics
            "collision_rate": np.mean(self.collisions),
            "success_rate": np.mean(self.success_episodes),
            "num_collisions": sum(self.collisions),
            "num_successes": sum(self.success_episodes),
            
            # Speed metrics
            "mean_speed": np.mean(self.speeds) if self.speeds else 0,
            "std_speed": np.std(self.speeds) if self.speeds else 0,
            "max_speed": np.max(self.speeds) if self.speeds else 0,
            
            # Behavior metrics
            "total_lane_changes": len(self.lane_changes),
            "total_zone_violations": len(self.construction_zone_violations),
        }
        
        return stats
    
    def print_summary(self):
        """Print evaluation summary"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nEpisode Statistics:")
        print(f"  Total episodes: {stats['num_episodes']}")
        print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Median reward: {stats['median_reward']:.2f}")
        print(f"  Min/Max reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"  Mean episode length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} steps")
        
        print(f"\nSafety Metrics:")
        print(f"  Success rate: {stats['success_rate']:.2%} ({stats['num_successes']}/{stats['num_episodes']})")
        print(f"  Collision rate: {stats['collision_rate']:.2%} ({stats['num_collisions']}/{stats['num_episodes']})")
        
        print(f"\nBehavior Metrics:")
        print(f"  Mean speed: {stats['mean_speed']:.2f} ± {stats['std_speed']:.2f}")
        print(f"  Max speed: {stats['max_speed']:.2f}")
        print(f"  Total lane changes: {stats['total_lane_changes']}")
        print(f"  Construction zone violations: {stats['total_zone_violations']}")
        print("=" * 80 + "\n")


def load_model(model_path: str, algorithm: str, env: gym.Env):
    """Load a trained model"""
    algorithm_class = {
        "dqn": DQN,
        "ppo": PPO,
        "sac": SAC,
        "a2c": A2C,
    }[algorithm.lower()]
    
    print(f"Loading {algorithm.upper()} model from: {model_path}")
    model = algorithm_class.load(model_path, env=env)
    return model


def create_env(env_id: str = "highway-with-obstacles-v0",
               config: Optional[Dict[str, Any]] = None,
               render_mode: Optional[str] = None) -> gym.Env:
    """Create and configure environment"""
    env = gym.make(env_id, render_mode=render_mode)
    
    if config:
        env.unwrapped.config.update(config)
    
    return env


def evaluate_model(
    model,
    env: gym.Env,
    n_eval_episodes: int = 100,
    deterministic: bool = True,
    verbose: bool = True,
) -> Tuple[EvaluationMetrics, List[List[float]]]:
    """
    Evaluate model with detailed metrics tracking
    
    Returns:
        metrics: EvaluationMetrics object with all tracked data
        episode_trajectories: List of reward trajectories for each episode
    """
    metrics = EvaluationMetrics()
    episode_trajectories = []
    
    if verbose:
        print(f"\nEvaluating model for {n_eval_episodes} episodes...")
        print("-" * 80)
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        episode_length = 0
        reward_trajectory = []
        
        last_lane_idx = None
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            reward_trajectory.append(reward)
            
            # Track step-level metrics
            if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
                vehicle = env.unwrapped.vehicle
                speed = vehicle.speed if hasattr(vehicle, 'speed') else 0
                
                # Detect lane change
                current_lane_idx = vehicle.lane_index if hasattr(vehicle, 'lane_index') else None
                lane_change = (last_lane_idx is not None and 
                              current_lane_idx is not None and 
                              last_lane_idx != current_lane_idx)
                last_lane_idx = current_lane_idx
                
                metrics.add_step_data(speed, lane_change=lane_change)
        
        metrics.add_episode(episode_reward, episode_length, info)
        episode_trajectories.append(reward_trajectory)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_eval_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}, "
                  f"Collision: {info.get('crashed', False)}")
    
    if verbose:
        print("-" * 80)
    
    return metrics, episode_trajectories


def visualize_results(
    metrics: EvaluationMetrics,
    episode_trajectories: List[List[float]],
    save_path: Path,
):
    """Create comprehensive visualization of evaluation results"""
    
    print(f"\nGenerating visualizations...")
    
    pdf_path = save_path / "evaluation_results.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: Reward distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Evaluation Results - Reward Analysis', fontsize=16, fontweight='bold')
        
        # Reward histogram
        axes[0, 0].hist(metrics.episode_rewards, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(metrics.episode_rewards), color='r', 
                          linestyle='--', label=f'Mean: {np.mean(metrics.episode_rewards):.2f}')
        axes[0, 0].axvline(np.median(metrics.episode_rewards), color='g', 
                          linestyle='--', label=f'Median: {np.median(metrics.episode_rewards):.2f}')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward over episodes
        axes[0, 1].plot(metrics.episode_rewards, alpha=0.6, linewidth=0.5)
        axes[0, 1].plot(np.convolve(metrics.episode_rewards, np.ones(10)/10, mode='valid'), 
                       color='red', linewidth=2, label='Moving Avg (10)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward over Episodes')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode length distribution
        axes[1, 0].hist(metrics.episode_lengths, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].axvline(np.mean(metrics.episode_lengths), color='r', 
                          linestyle='--', label=f'Mean: {np.mean(metrics.episode_lengths):.1f}')
        axes[1, 0].set_xlabel('Episode Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Episode Length Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success vs Collision pie chart
        stats = metrics.get_statistics()
        labels = ['Success', 'Collision']
        sizes = [stats['num_successes'], stats['num_collisions']]
        colors = ['#90EE90', '#FF6B6B']
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                      startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        axes[1, 1].set_title('Episode Outcomes')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Speed and behavior analysis
        if metrics.speeds:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Evaluation Results - Behavior Analysis', fontsize=16, fontweight='bold')
            
            # Speed distribution
            axes[0, 0].hist(metrics.speeds, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            axes[0, 0].axvline(np.mean(metrics.speeds), color='r', 
                              linestyle='--', label=f'Mean: {np.mean(metrics.speeds):.2f}')
            axes[0, 0].set_xlabel('Speed')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Speed Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Reward trajectory for first few episodes
            axes[0, 1].set_title('Reward Trajectories (First 5 Episodes)')
            for i, traj in enumerate(episode_trajectories[:5]):
                axes[0, 1].plot(traj, label=f'Episode {i+1}', alpha=0.7)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Metrics summary table
            axes[1, 0].axis('off')
            summary_data = [
                ['Metric', 'Value'],
                ['Mean Reward', f"{stats['mean_reward']:.2f}"],
                ['Success Rate', f"{stats['success_rate']:.2%}"],
                ['Collision Rate', f"{stats['collision_rate']:.2%}"],
                ['Mean Speed', f"{stats['mean_speed']:.2f}"],
                ['Mean Length', f"{stats['mean_length']:.1f}"],
                ['Lane Changes', f"{stats['total_lane_changes']}"],
            ]
            table = axes[1, 0].table(cellText=summary_data, cellLoc='left',
                                    loc='center', colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            # Style header row
            for i in range(2):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            axes[1, 0].set_title('Performance Summary', pad=20, fontweight='bold')
            
            # Cumulative reward
            cumulative_rewards = np.cumsum(metrics.episode_rewards)
            axes[1, 1].plot(cumulative_rewards, linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Cumulative Reward')
            axes[1, 1].set_title('Cumulative Reward over Episodes')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Page 3: Violin plot for performance test
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle('Performance Test: Mean Episodic Return Distribution', fontsize=16, fontweight='bold')
        
        # Create violin plot
        parts = ax.violinplot([metrics.episode_rewards], positions=[1], widths=0.7,
                              showmeans=True, showmedians=True, showextrema=True)
        
        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor('#4CAF50')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Customize other elements
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('blue')
        parts['cmedians'].set_linewidth(2)
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        
        # Add scatter points with jitter for individual episodes
        np.random.seed(42)
        jitter = np.random.normal(0, 0.02, size=len(metrics.episode_rewards))
        ax.scatter(np.ones(len(metrics.episode_rewards)) + jitter, 
                  metrics.episode_rewards, alpha=0.3, s=20, color='darkgreen', 
                  label='Individual Episodes')
        
        # Add statistical annotations
        mean_val = np.mean(metrics.episode_rewards)
        median_val = np.median(metrics.episode_rewards)
        std_val = np.std(metrics.episode_rewards)
        
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axhline(y=median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.2f}')
        
        # Add text annotations
        text_str = f'Statistics (n={len(metrics.episode_rewards)} episodes):\\n'
        text_str += f'Mean: {mean_val:.2f}\\n'
        text_str += f'Std: {std_val:.2f}\\n'
        text_str += f'Median: {median_val:.2f}\\n'
        text_str += f'Min: {np.min(metrics.episode_rewards):.2f}\\n'
        text_str += f'Max: {np.max(metrics.episode_rewards):.2f}\\n'
        text_str += f'Q1: {np.percentile(metrics.episode_rewards, 25):.2f}\\n'
        text_str += f'Q3: {np.percentile(metrics.episode_rewards, 75):.2f}'
        
        ax.text(1.35, np.mean(metrics.episode_rewards), text_str,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, verticalalignment='center')
        
        ax.set_ylabel('Mean Episodic Return (Reward)', fontsize=12)
        ax.set_title(f'Performance Test Results (Deterministic Policy, {len(metrics.episode_rewards)} Episodes)',
                    fontsize=12, pad=15)
        ax.set_xticks([1])
        ax.set_xticklabels(['Model Performance'])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"Visualizations saved to: {pdf_path}")
    
    # Also save standalone violin plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    parts = ax.violinplot([metrics.episode_rewards], positions=[1], widths=0.7,
                          showmeans=True, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4CAF50')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('blue')
    parts['cmedians'].set_linewidth(2)
    
    np.random.seed(42)
    jitter = np.random.normal(0, 0.02, size=len(metrics.episode_rewards))
    ax.scatter(np.ones(len(metrics.episode_rewards)) + jitter, 
              metrics.episode_rewards, alpha=0.3, s=20, color='darkgreen')
    
    mean_val = np.mean(metrics.episode_rewards)
    median_val = np.median(metrics.episode_rewards)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.2f}')
    ax.axhline(y=median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.2f}')
    
    ax.set_ylabel('Mean Episodic Return (Reward)', fontsize=14, fontweight='bold')
    ax.set_title(f'Performance Test: Mean Episodic Return\\n(Deterministic Policy, {len(metrics.episode_rewards)} Episodes)',
                fontsize=14, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['Trained Model'])
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    violin_path = save_path / 'violin_plot_performance.png'
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved to: {violin_path}")


def record_videos(
    model,
    env_config: Optional[Dict[str, Any]],
    video_folder: Path,
    n_videos: int = 5,
    video_length: int = 1000,
    deterministic: bool = True,
    algorithm: str = "ppo",
):
    """Record videos of the agent's performance"""
    
    print(f"\nRecording {n_videos} videos...")
    video_folder.mkdir(parents=True, exist_ok=True)
    
    # Create environment with video recording
    env = create_env("highway-with-obstacles-v0", env_config, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_folder),
        episode_trigger=lambda x: True,
        name_prefix=f"{algorithm}_eval"
    )
    env.unwrapped.config["simulation_frequency"] = 15
    
    for video_idx in range(n_videos):
        obs, info = env.reset()
        done = truncated = False
        steps = 0
        episode_reward = 0
        
        print(f"  Recording video {video_idx + 1}/{n_videos}...", end=" ")
        
        while not (done or truncated) and steps < video_length:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        print(f"Done! Reward: {episode_reward:.2f}, Steps: {steps}, "
              f"Collision: {info.get('crashed', False)}")
    
    env.close()
    print(f"Videos saved to: {video_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agents on Highway Environment"
    )
    
    # Model settings
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="ppo",
        choices=["dqn", "ppo", "sac", "a2c"],
        help="RL algorithm used"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--n-episodes", "-n",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy"
    )
    
    # Video settings
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos of agent performance"
    )
    
    parser.add_argument(
        "--n-videos",
        type=int,
        default=5,
        help="Number of videos to record"
    )
    
    parser.add_argument(
        "--video-length",
        type=int,
        default=1000,
        help="Maximum length of each video (steps)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom environment config JSON file"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).stem
    output_path = Path(args.output_dir) / f"{model_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Number of episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    
    # Load environment config
    env_config = None
    if args.config:
        with open(args.config, "r") as f:
            env_config = json.load(f)
        print(f"Loaded custom config from: {args.config}")
    
    # Create environment
    env = create_env("highway-with-obstacles-v0", env_config, render_mode=None)
    
    # Load model
    model = load_model(args.model_path, args.algorithm, env)
    
    # Evaluate model
    metrics, episode_trajectories = evaluate_model(
        model,
        env,
        n_eval_episodes=args.n_episodes,
        deterministic=args.deterministic,
        verbose=args.verbose,
    )
    
    # Print summary
    metrics.print_summary()
    
    # Save statistics to JSON
    stats = metrics.get_statistics()
    stats_path = output_path / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    # Save detailed results
    results_path = output_path / "detailed_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "episode_rewards": metrics.episode_rewards,
            "episode_lengths": metrics.episode_lengths,
            "collisions": metrics.collisions,
            "success_episodes": metrics.success_episodes,
        }, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    # Generate visualizations
    if not args.no_visualize:
        visualize_results(metrics, episode_trajectories, output_path)
    
    # Record videos
    if args.record_video:
        video_folder = output_path / "videos"
        record_videos(
            model,
            env_config,
            video_folder,
            n_videos=args.n_videos,
            video_length=args.video_length,
            deterministic=args.deterministic,
            algorithm=args.algorithm,
        )
    
    env.close()
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print(f"All results saved to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
