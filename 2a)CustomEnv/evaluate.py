"""
Evaluation Script for Trained Models

Test and visualize trained agents in the NarrowLaneSafeChange-v0 environment.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
import sys
import time

# Import custom environment
import __init__


def load_and_evaluate(
    model_path: str,
    algorithm: str = "dqn",
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
):
    """
    Load a trained model and evaluate its performance.
    
    Args:
        model_path: Path to the saved model
        algorithm: Algorithm type ("dqn" or "ppo")
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
    """
    print(f"Loading {algorithm.upper()} model from {model_path}...")
    
    # Load the model
    if algorithm.lower() == "dqn":
        model = DQN.load(model_path)
    elif algorithm.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make('NarrowLaneSafeChange-v0', render_mode=render_mode)
    
    # Evaluate the model
    print(f"\nEvaluating model for {n_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        render=render
    )
    
    print(f"\nResults:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return mean_reward, std_reward


def run_episodes_with_stats(
    model_path: str,
    algorithm: str = "dqn",
    n_episodes: int = 5,
    render: bool = False
):
    """
    Run episodes and collect detailed statistics.
    
    Returns episode rewards, lengths, crashes, and lane changes.
    """
    print(f"Loading {algorithm.upper()} model from {model_path}...")
    
    # Load the model
    if algorithm.lower() == "dqn":
        model = DQN.load(model_path)
    elif algorithm.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make('NarrowLaneSafeChange-v0', render_mode=render_mode)
    
    # Collect statistics
    episode_rewards = []
    episode_lengths = []
    episode_crashes = []
    episode_lane_changes = []
    episode_speeds = []
    
    print(f"\nRunning {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        last_lane = info.get('lane', 0)
        lane_changes = 0
        speeds = []
        
        while not (done or truncated):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track lane changes
            current_lane = info.get('lane', 0)
            if current_lane != last_lane:
                lane_changes += 1
            last_lane = current_lane
            
            # Track speed
            speeds.append(info.get('speed', 0))
            
            if render:
                env.render()
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_crashes.append(1 if info.get('crashed', False) else 0)
        episode_lane_changes.append(lane_changes)
        episode_speeds.append(np.mean(speeds))
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Crashed={info.get('crashed', False)}, "
              f"Lane Changes={lane_changes}, "
              f"Avg Speed={np.mean(speeds):.2f} m/s")
    
    env.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Crash Rate: {np.mean(episode_crashes) * 100:.1f}%")
    print(f"Average Lane Changes: {np.mean(episode_lane_changes):.1f} ± {np.std(episode_lane_changes):.1f}")
    print(f"Average Speed: {np.mean(episode_speeds):.2f} ± {np.std(episode_speeds):.2f} m/s")
    print("=" * 60)
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'crashes': episode_crashes,
        'lane_changes': episode_lane_changes,
        'speeds': episode_speeds
    }


def compare_models(model_paths: dict, n_episodes: int = 10):
    """
    Compare multiple trained models.
    
    Args:
        model_paths: Dictionary mapping model names to (path, algorithm) tuples
        n_episodes: Number of episodes to evaluate each model
    """
    results = {}
    
    for model_name, (model_path, algorithm) in model_paths.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 60}")
        
        stats = run_episodes_with_stats(
            model_path=model_path,
            algorithm=algorithm,
            n_episodes=n_episodes,
            render=False
        )
        results[model_name] = stats
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    
    metrics = [
        ('rewards', 'Episode Rewards', axes[0, 0]),
        ('lengths', 'Episode Lengths', axes[0, 1]),
        ('lane_changes', 'Lane Changes per Episode', axes[1, 0]),
        ('speeds', 'Average Speed (m/s)', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        for model_name, stats in results.items():
            ax.hist(stats[metric], alpha=0.6, label=model_name, bins=10)
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to 'model_comparison.png'")
    plt.show()
    
    return results


def watch_agent(model_path: str, algorithm: str = "dqn", n_episodes: int = 3, fps: int = 15):
    """
    Watch the agent play with pygame rendering enabled.
    
    Args:
        model_path: Path to the saved model
        algorithm: Algorithm type ("dqn" or "ppo")
        n_episodes: Number of episodes to watch
        fps: Frames per second for rendering
    """
    print(f"Watching {algorithm.upper()} agent with pygame visualization...")
    print(f"Close the pygame window or press Ctrl+C to stop.\n")
    
    # Load the model
    if algorithm.lower() == "dqn":
        model = DQN.load(model_path)
    elif algorithm.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create environment with pygame rendering
    env = gym.make('NarrowLaneSafeChange-v0', render_mode='human')
    env.unwrapped.config['show_trajectories'] = True
    
    frame_delay = 1.0 / fps
    
    try:
        for episode in range(n_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*60}")
            
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            last_lane = info.get('lane', 0)
            lane_changes = 0
            speeds = []
            
            while not (done or truncated):
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track lane changes
                current_lane = info.get('lane', 0)
                if current_lane != last_lane:
                    lane_changes += 1
                    print(f"  Lane change at step {episode_length}: Lane {last_lane} → Lane {current_lane}")
                last_lane = current_lane
                
                # Track speed
                speeds.append(info.get('speed', 0))
                
                # Render and control frame rate
                env.render()
                time.sleep(frame_delay)
                
                if done or truncated:
                    crash_msg = "CRASHED" if info.get('crashed', False) else "Completed"
                    print(f"\n  Episode {crash_msg}!")
                    print(f"  Length: {episode_length} steps")
                    print(f"  Total Reward: {episode_reward:.2f}")
                    print(f"  Lane Changes: {lane_changes}")
                    print(f"  Avg Speed: {np.mean(speeds):.2f} m/s")
                    time.sleep(2)  # Pause before next episode
                    break
    
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    finally:
        env.close()
        print("\nVisualization ended.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "ppo"],
        help="Algorithm type"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Watch agent with pygame visualization (interactive)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for rendering (default: 15)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path + ".zip"):
        print(f"Error: Model not found at {args.model_path}")
        print("Make sure to provide the path without the .zip extension")
        sys.exit(1)
    
    # Use pygame visualization if render flag is set
    if args.render:
        watch_agent(
            model_path=args.model_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes,
            fps=args.fps
        )
    elif args.stats:
        run_episodes_with_stats(
            model_path=args.model_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes,
            render=False
        )
    else:
        load_and_evaluate(
            model_path=args.model_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes,
            render=False
        )
