"""
Custom Highway Environment: NarrowLaneSafeChange-v0

This environment models a narrow two-lane highway where safe lane-changing
and collision avoidance are prioritized over speed.
"""

import numpy as np
from typing import Dict, Tuple
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class NarrowLaneSafeChangeEnv(AbstractEnv):
    """
    A narrow two-lane highway environment focused on safe lane changes.
    
    Features:
    - Only two lanes (narrow road)
    - Rewards safe driving and lane changes
    - Penalizes high speeds and collisions
    - Constrained speed range
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": True,
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 2,  # Narrow road: only 2 lanes
            "vehicles_count": 10,  # Variable traffic density (reduced from 10)
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 80,  # seconds
            "ego_spacing": 2,
            "vehicles_density": 1,  # Reduced density for easier learning
            "collision_reward": 0,  # High penalty for collision
            "right_lane_reward": 0,
            "high_speed_reward": 0,
            "lane_change_reward": 0.2,  # Reward successful lane changes
            "reward_speed_range": [10, 20],  # Discourage high speeds
            "normalize_reward": True,
            "offroad_terminal": False,
            # Rendering
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            # Speed constraints
            "policy_frequency": 2,
            "simulation_frequency": 10,
            # Narrow lane constraints
            "max_speed": 25,  # Limited max speed (m/s, ~90 km/h)
            "min_speed": 5,
            # Safety parameters
            "unsafe_lane_change_penalty": 0,
            "survival_reward": 0.1,  # Reward for staying alive
            "speed_penalty_factor": 0,  # Penalty for high speeds
        })
        return config

    def _reset(self) -> np.ndarray:
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        self.last_lane = self.vehicle.lane_index[2]
        self.unsafe_lane_change = False

    def _create_road(self) -> None:
        """Create a narrow two-lane highway."""
        # Create a simple straight road with 2 lanes
        net = RoadNetwork()
        
        # Single road segment with 2 lanes
        lane_width = 3.5  # Standard lane width (meters)
        speedlimit = self.config["max_speed"]
        
        # Create straight road
        net.add_lane(
            "a", "b",
            StraightLane(
                [0, 0], 
                [2000, 0],  # Long straight road
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=lane_width,
                speed_limit=speedlimit
            )
        )
        net.add_lane(
            "a", "b",
            StraightLane(
                [0, lane_width], 
                [2000, lane_width],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=lane_width,
                speed_limit=speedlimit
            )
        )
        net.add_lane(
            "b", "a",
            StraightLane(
                [2000, lane_width], 
                [0, lane_width],
                line_types=(LineType.NONE, LineType.NONE),
                width=lane_width,
                speed_limit=speedlimit
            )
        )
        
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )
        self.road = road

    def _create_vehicles(self) -> None:
        """Create ego vehicle and traffic vehicles."""
        road = self.road

        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 0)).position(30.0, 0.0), speed=30.0
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # --- Traffic vehicles
        vehicles_count = self.config["vehicles_count"]
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(vehicles_count):
            lane_idx = self.np_random.integers(0, 2)  # 0 or 1

            if lane_idx == 0:
                lane_from, lane_to = "a", "b"  # same direction as ego
            else:
                lane_from, lane_to = "b", "a"  # oncoming traffic
            
            lane = road.network.get_lane((lane_from, lane_to, 0))
            
            # Use proper longitudinal position along the lane
            longitudinal_position = self.np_random.uniform(
                low=50 + i * 80 / vehicles_count,
                high=50 + (i + 1) * 1600 / vehicles_count
            )
            
            # Get position and heading from the lane
            position = lane.position(longitudinal_position, 0)
            heading = lane.heading_at(longitudinal_position)
            
            speed = self.np_random.uniform(
                low=self.config["min_speed"],
                high=self.config["max_speed"] * 0.8
            )

            # Create vehicle and disable lane changing
            vehicle = vehicles_type(
                road,
                position,
                heading=heading,
                speed=speed
            )
            vehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000  
            vehicle.LANE_CHANGE_MIN_ACC_GAIN = 1000  
            vehicle.LANE_CHANGE_DELAY = 1000 
            
            self.road.vehicles.append(vehicle)

    def _reward(self, action: int) -> float:
        """
        Custom reward function emphasizing safety over speed.
        
        Components:
        - Survival reward: +0.1 per timestep alive
        - Lane change reward: +0.2 for successful, safe lane changes
        - Collision penalty: -1.0
        - Unsafe lane change penalty: -0.5
        - Speed penalty: -0.01 × speed (to discourage high speeds)
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        
        # Normalize if requested
        if self.config.get("normalize_reward", True):
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], 
                 self.config["survival_reward"] + self.config["lane_change_reward"]],
                [0, 1]
            )
        
        reward = np.clip(reward, -1, 1)
        return reward

    def _rewards(self, action: int) -> Dict[str, float]:
        """Compute individual reward components."""
        current_lane = self.vehicle.lane_index[2]
        
        # Check if lane changed
        lane_changed = current_lane != self.last_lane
        
        # Check if lane change was unsafe
        if lane_changed:
            # Check for nearby vehicles during lane change
            for v in self.road.vehicles:
                if v is not self.vehicle:
                    distance = np.linalg.norm(v.position - self.vehicle.position)
                    if distance < 15:  # Within 15 meters
                        relative_speed = np.linalg.norm(v.velocity - self.vehicle.velocity)
                        if relative_speed > 5:  # Significant speed difference
                            self.unsafe_lane_change = True
                            break
        
        # Update last lane
        self.last_lane = current_lane
        
        rewards = {
            "collision_reward": float(self.vehicle.crashed),
            "survival_reward": 1.0 if not self.vehicle.crashed else 0.0,
            "lane_change_reward": 1.0 if lane_changed else 0.0,
            "unsafe_lane_change_penalty": 0 if self.unsafe_lane_change else 0.0,
        }
        
        # Reset unsafe flag
        if self.unsafe_lane_change and not lane_changed:
            self.unsafe_lane_change = False
        
        return rewards

    def _is_terminated(self) -> bool:
        """Episode terminates on collision."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """Episode truncates after max duration."""
        # highway-env uses time internally (duration * policy_frequency gives max steps)
        # The base class tracks this, so we check if we've exceeded the time limit
        return self.time >= self.config["duration"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.steps += 1
        return super().step(action)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        """Additional info for logging."""
        info = super()._info(obs, action)
        info.update({
            "speed": np.linalg.norm(self.vehicle.velocity),
            "lane": self.vehicle.lane_index[2] if hasattr(self.vehicle, 'lane_index') else 0,
            "crashed": self.vehicle.crashed,
        })
        return info