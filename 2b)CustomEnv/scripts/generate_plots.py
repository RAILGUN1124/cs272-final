#!/usr/bin/env python3
"""
Generate Required Plots Script
Creates the two specific plots needed:
1. Learning curve (Mean episodic training reward vs. training episodes)
2. Violin plot (Mean episodic training reward for evaluation episodes)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curve_from_logs(log_dir: Path, output_path: Path):
    """
    Generate learning curve from training logs
    
    Args:
        log_dir: Directory containing progress.csv from training
        output_path: Where to save the plot
    """
    # Try to read from progress.csv
    progress_file = log_dir / "progress.csv"
    
    if not progress_file.exists():
        # Try to find evaluations.npz
        eval_file = log_dir / "evaluations.npz"
        if eval_file.exists():
            data = np.load(eval_file)
            episodes = data['timesteps'] if 'timesteps' in data else np.arange(len(data['results']))
            rewards = data['results']
            
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(episodes, rewards, linewidth=2, color='#2E86DE', marker='o', markersize=4)
            
            ax.set_xlabel('Training Timesteps', fontsize=14, fontweight='bold')
            ax.set_ylabel('Mean Episodic Return (Reward)', fontsize=14, fontweight='bold')
            ax.set_title('Learning Curve: Training Progress', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics box
            mean_reward = np.mean(rewards)
            final_reward = rewards[-1] if len(rewards) > 0 else 0
            max_reward = np.max(rewards)
            
            stats_text = f'Final: {final_reward:.2f}\\nMean: {mean_reward:.2f}\\nMax: {max_reward:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Learning curve saved to: {output_path}")
            return
    
    # Read from progress.csv
    try:
        import pandas as pd
        df = pd.read_csv(progress_file)
        
        # Try different column names
        reward_col = None
        episode_col = None
        
        for col in ['rollout/ep_rew_mean', 'ep_rew_mean', 'mean_reward']:
            if col in df.columns:
                reward_col = col
                break
        
        for col in ['total_timesteps', 'timesteps', 'time/total_timesteps']:
            if col in df.columns:
                episode_col = col
                break
        
        if reward_col and episode_col:
            episodes = df[episode_col].values
            rewards = df[reward_col].values
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot raw data
            ax.plot(episodes, rewards, alpha=0.4, linewidth=1, color='lightblue', label='Training Data')
            
            # Plot smoothed curve
            if len(rewards) >= 10:
                window = min(len(rewards) // 10, 50)
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                smooth_episodes = episodes[window-1:]
                ax.plot(smooth_episodes, smoothed, linewidth=3, color='#2E86DE', 
                       label=f'Smoothed (window={window})')
            
            ax.set_xlabel('Training Timesteps', fontsize=14, fontweight='bold')
            ax.set_ylabel('Mean Episodic Return (Reward)', fontsize=14, fontweight='bold')
            ax.set_title('Learning Curve: Training Progress', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics
            mean_reward = np.mean(rewards)
            final_reward = rewards[-1] if len(rewards) > 0 else 0
            max_reward = np.max(rewards)
            
            stats_text = f'Final: {final_reward:.2f}\\nMean: {mean_reward:.2f}\\nMax: {max_reward:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Learning curve saved to: {output_path}")
        else:
            print(f"✗ Could not find reward columns in {progress_file}")
            print(f"Available columns: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"✗ Error reading progress file: {e}")


def plot_violin_from_results(results_file: Path, output_path: Path, n_episodes: int = None):
    """
    Generate violin plot from evaluation results
    
    Args:
        results_file: JSON file with detailed results
        output_path: Where to save the plot
        n_episodes: Number of episodes evaluated
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data.get('episode_rewards', [])
    
    if not episode_rewards:
        print(f"✗ No episode rewards found in {results_file}")
        return
    
    n_episodes = n_episodes or len(episode_rewards)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create violin plot
    parts = ax.violinplot([episode_rewards], positions=[1], widths=0.7,
                          showmeans=True, showmedians=True, showextrema=True)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('#4CAF50')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(2)
    
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(3)
    parts['cmeans'].set_label('Mean')
    parts['cmedians'].set_color('blue')
    parts['cmedians'].set_linewidth(3)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(1.5)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(1.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(1.5)
    
    # Add scatter points with jitter
    np.random.seed(42)
    jitter = np.random.normal(0, 0.02, size=len(episode_rewards))
    ax.scatter(np.ones(len(episode_rewards)) + jitter, 
              episode_rewards, alpha=0.3, s=30, color='darkgreen',
              edgecolors='black', linewidths=0.5, label='Individual Episodes')
    
    # Add statistical lines
    mean_val = np.mean(episode_rewards)
    median_val = np.median(episode_rewards)
    std_val = np.std(episode_rewards)
    
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8, 
              label=f'Mean: {mean_val:.2f}')
    ax.axhline(y=median_val, color='blue', linestyle='--', linewidth=2, alpha=0.8,
              label=f'Median: {median_val:.2f}')
    
    # Add statistics text box
    text_str = f'Statistics (n={len(episode_rewards)}):\\n'
    text_str += f'───────────────\\n'
    text_str += f'Mean:    {mean_val:.2f}\\n'
    text_str += f'Std:     {std_val:.2f}\\n'
    text_str += f'Median:  {median_val:.2f}\\n'
    text_str += f'Min:     {np.min(episode_rewards):.2f}\\n'
    text_str += f'Max:     {np.max(episode_rewards):.2f}\\n'
    text_str += f'Q1:      {np.percentile(episode_rewards, 25):.2f}\\n'
    text_str += f'Q3:      {np.percentile(episode_rewards, 75):.2f}'
    
    ax.text(1.35, np.median(episode_rewards), text_str,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
           fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    ax.set_ylabel('Mean Episodic Return (Reward)', fontsize=14, fontweight='bold')
    ax.set_title(f'Performance Test: Mean Episodic Return Distribution\\n' + 
                f'Deterministic Policy Evaluation ({n_episodes} Episodes)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([1])
    ax.set_xticklabels(['Trained Model'], fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Set reasonable y-axis limits
    y_min = min(episode_rewards) - 0.1 * abs(min(episode_rewards))
    y_max = max(episode_rewards) + 0.1 * abs(max(episode_rewards))
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Violin plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate required plots for RL training analysis"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        help="Directory containing training logs (for learning curve)"
    )
    
    parser.add_argument(
        "--results-file", "-r",
        type=str,
        help="JSON file with evaluation results (for violin plot)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="plots",
        help="Directory to save plots"
    )
    
    parser.add_argument(
        "--learning-curve-only",
        action="store_true",
        help="Only generate learning curve"
    )
    
    parser.add_argument(
        "--violin-only",
        action="store_true",
        help="Only generate violin plot"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  Generating Required Plots")
    print("=" * 80)
    
    # Generate learning curve
    if not args.violin_only:
        if args.log_dir:
            log_dir = Path(args.log_dir)
            output_path = output_dir / "learning_curve.png"
            print(f"\\nGenerating learning curve from: {log_dir}")
            plot_learning_curve_from_logs(log_dir, output_path)
        else:
            print("\\n✗ No log directory provided for learning curve (use --log-dir)")
    
    # Generate violin plot
    if not args.learning_curve_only:
        if args.results_file:
            results_file = Path(args.results_file)
            output_path = output_dir / "violin_plot_performance.png"
            print(f"\\nGenerating violin plot from: {results_file}")
            plot_violin_from_results(results_file, output_path)
        else:
            print("\\n✗ No results file provided for violin plot (use --results-file)")
    
    print("\\n" + "=" * 80)
    print(f"Plots saved to: {output_dir}")
    print("=" * 80 + "\\n")


if __name__ == "__main__":
    main()
