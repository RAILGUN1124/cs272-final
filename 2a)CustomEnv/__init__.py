"""
Custom Highway Environment Package

Register custom environments with Gymnasium.
"""

from gymnasium.envs.registration import register

from narrow_lane_env import NarrowLaneSafeChangeEnv

# Register the custom environment
register(
    id='NarrowLaneSafeChange-v0',
    entry_point='narrow_lane_env:NarrowLaneSafeChangeEnv',
    max_episode_steps=200,
)

__all__ = ['NarrowLaneSafeChangeEnv']
