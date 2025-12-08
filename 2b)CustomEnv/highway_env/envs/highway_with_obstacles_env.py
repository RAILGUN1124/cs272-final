from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class HighwayWithObstaclesEnv(HighwayEnv):
    """
    A highway driving environment with static obstacles and construction zones on the road.

    The vehicle is driving on a straight highway with several lanes and static obstacles,
    and is rewarded for reaching a high speed, staying on the rightmost lanes and avoiding collisions.
    also should be rewarded for keeping speed limits within construction zones and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "obstacles_count": 10,  # Number of static obstacles on the road
                "obstacle_spacing": 20,  # Minimum spacing between obstacles [m]
                "obstacle_on_lanes": None,  # List of lane indices where obstacles can appear, None for all lanes
                 # Construction zone configurations
                "construction_zones_count": 2,  # Number of construction zones
                "construction_zone_length": 100,  # Length of each zone [m]
                "construction_zone_side": "random",  # "left", "right", or "random"
                "construction_zone_lanes": 2,  # Number of lanes the zone takes up
                "construction_cone_spacing": 5,  # Distance between cones [m]

                "reward": {
                    "collision_penalty": -1.0,
                    "closed_lane_penalty": -1.0,
                    "progress_reward": {
                    "type": "percentage",
                    "description": "Distance covered as percentage (e.g., 0.73 if 73% covered)"
                    },
                    "speed_compliance": {
                    "within_limit": 0.05,
                    },
                    "speed_violation": {
                    "beyond_limit": -0.05,
                    }
                },

                "speed": {
                    "construction_zone_limit_mph": 45,
                    "construction_zone_limit_kmh": 72.42,
                    "speed_tolerance_mph": 5,
                    "speed_tolerance_kmh": 8.05,
                    "description": "Must maintain speed within ±5 mph of construction zone limit"
                },

                "safety_rules": {
                    "collision": {
                    "penalty": -1.0,
                    },
                    "closed_lane": {
                    "penalty": -1.0,
                    }
                },
            }
        )
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self.construction_zones = []  # Store zone boundaries for vehicle placement checks
        self._create_construction_zones()
        self._create_obstacles()
        self._create_vehicles()


    def _create_construction_zones(self) -> None:
        """
        Create construction zones on the highway with traffic cones marking the boundaries.
        Zones start from left or right edge and taper towards center.
        """
        zones_count = self.config["construction_zones_count"]
        zone_length = self.config["construction_zone_length"]
        zone_side = self.config["construction_zone_side"]
        zone_lanes = self.config["construction_zone_lanes"]
        cone_spacing = self.config["construction_cone_spacing"]
        lanes_count = self.config["lanes_count"]
        
        
        if zones_count == 0:
            return
        
        zone_lanes = min(zone_lanes, lanes_count - 1)
        
        network = self.road.network
        lane_index = ("0", "1", 0)
        lane = network.get_lane(lane_index)
        
        ego_start = 100  # Approximate ego spawn position
        visible_start = ego_start + 300  # Start zones 300m ahead of ego
        usable_length = lane.length - visible_start - 100  # Keep buffer at end
        
        # create each construction zone
        for zone_idx in range(zones_count):

            if zone_side == "random":
                side = self.np_random.choice(["left", "right"])
            else:
                side = zone_side
            
            if zones_count == 1:
                start_long = visible_start + usable_length / 2
            else:
                start_long = visible_start + (zone_idx * (usable_length - zone_length) / (zones_count - 1))
            
            end_long = start_long + zone_length
            
            # Store zone boundaries for vehicle placement checks
            self.construction_zones.append({
                'start': start_long,
                'end': end_long,
                'side': side,
                'lanes': zone_lanes
            })
            
            # Create the tapered construction zone
            self._create_zone_with_cones(
                start_long, end_long, side, zone_lanes, cone_spacing
            )
        
    def _create_zone_with_cones(
        self, 
        start_long: float, 
        end_long: float, 
        side: str, 
        num_lanes: int, 
        cone_spacing: float
    ) -> None:
        """
        Create a construction zone with cones forming a tapered boundary.
        
        :param start_long: Starting longitudinal position [m]
        :param end_long: Ending longitudinal position [m]
        :param side: "left" or "right" - which side of road zone starts from
        :param num_lanes: Number of lanes the zone occupies at maximum
        :param cone_spacing: Distance between consecutive cones [m]
        """
        network = self.road.network
        lanes_count = self.config["lanes_count"]
        
        # Get a reference lane for positioning
        reference_lane_idx = lanes_count - 1 if side == "left" else 0
        reference_lane = network.get_lane(("0", "1", reference_lane_idx))
        lane_width = reference_lane.width_at(start_long)
         
        num_cones = int((end_long - start_long) / cone_spacing)
        
        for i in range(num_cones + 1):
            # Current longitudinal position
            long_pos = start_long + i * cone_spacing
            if long_pos > end_long:
                break
            
            progress = (long_pos - start_long) / (end_long - start_long)
            
            taper_in_end = 0.3
            taper_out_start = 0.7
            
            if progress <= taper_in_end:
                taper_progress = progress / taper_in_end
            elif progress <= taper_out_start:
                taper_progress = 1.0
            else:
                taper_progress = (1 - progress) / (1 - taper_out_start)
            
            # Calculate lateral offset for the taper
            max_lateral_shift = num_lanes * lane_width
            current_lateral_shift = taper_progress * max_lateral_shift
            
            # Use the outermost affected lane as reference
            if side == "left":
                # Left side: start from leftmost lane, move right (negative lateral offset)
                cone_lane_idx = lanes_count - 1  # Leftmost lane
                lateral_offset = lane_width / 2 - current_lateral_shift  # Start at left edge, move inward
            else:  # right
                # Right side: start from rightmost lane, move left (positive lateral offset)
                cone_lane_idx = 0  # Rightmost lane
                lateral_offset = -lane_width / 2 + current_lateral_shift  # Start at right edge, move inward
            
            # Get the lane for positioning
            lane_index = ("0", "1", cone_lane_idx)
            lane_obj = network.get_lane(lane_index)
            
            position = lane_obj.position(long_pos, lateral_offset)
            heading = lane_obj.heading_at(long_pos)
            
            # Create traffic cone
            cone = Obstacle(self.road, position, heading=heading, speed=0)
            cone.LENGTH = 2.0
            cone.WIDTH = 2.0
            cone.collidable = True
            self.road.objects.append(cone)
            
       
    def _create_obstacles(self) -> None:
        """Create static obstacles on the road."""
        obstacles_count = self.config["obstacles_count"]
        obstacle_spacing = self.config["obstacle_spacing"]
        obstacle_lanes = self.config["obstacle_on_lanes"]
        
        if obstacle_lanes is None:
            obstacle_lanes = list(range(self.config["lanes_count"]))
        
        network = self.road.network
        lane_index = ("0", "1", 0)  # First lane of the straight road
        lane = network.get_lane(lane_index) # for getting total length
        
        # Create obstacles at random positions
        for _ in range(obstacles_count):
            longitudinal = self.np_random.uniform(50, lane.length - 50)
            
            lane_id = self.np_random.choice(obstacle_lanes)
            obstacle_lane_index = ("0", "1", lane_id)
            
            obstacle_lane = network.get_lane(obstacle_lane_index)
            position = obstacle_lane.position(longitudinal, 0)
            heading = obstacle_lane.heading_at(longitudinal)
            
            if self._is_position_safe(position, obstacle_spacing):
                obstacle = Obstacle(self.road, position, heading=heading, speed=0)
                self.road.objects.append(obstacle)

    def _is_position_safe(self, position: np.ndarray, min_distance: float) -> bool:
        """
        Check if a position is safe (not too close to existing vehicles or obstacles).
        
        :param position: position to check [x, y]
        :param min_distance: minimum safe distance [m]
        :return: True if position is safe
        """
        # Check distance to all vehicles
        for vehicle in self.road.vehicles:
            if np.linalg.norm(vehicle.position - position) < min_distance: # checks euclidian distance
                return False
        
        # Check distance to existing obstacles
        for obj in self.road.objects:
            if np.linalg.norm(obj.position - position) < min_distance:
                return False
        
        return True

    def _is_in_forbidden_construction_zone(self, longitudinal_pos: float, lane_idx: int) -> bool:
        """
        Check if a position is inside the area that is bounded by construction zone obstacles.
        
        :param longitudinal_pos: longitudinal position along road [m]
        :param lane_idx: lane index (0=rightmost, lanes_count-1=leftmost)
        :return: True if position is inside a construction zone
        """
        if not hasattr(self, 'construction_zones'):
            return False
            
        lanes_count = self.config["lanes_count"]
        buffer = 20  # Additional buffer zone [m] - increased for safety
        
        for zone in self.construction_zones:
            # Check if longitudinal position is within zone (with buffer)
            if zone['start'] - buffer <= longitudinal_pos <= zone['end'] + buffer:
                # Determine affected lanes based on zone side
                # Add +1 to affected lanes to include adjacent lane for extra safety
                if zone['side'] == 'left':
                    affected_lanes = list(range(max(0, lanes_count - zone['lanes'] - 1), lanes_count))
                else:  # right
                    affected_lanes = list(range(0, min(lanes_count, zone['lanes'] + 1)))
                
                # Check if lane is affected
                if lane_idx in affected_lanes:
                    return True
        
        return False

    def _is_in_construction_zone(self, longitudinal_pos: float) -> bool:
        """
        Check if a position is inside a construction zone.
        
        :param longitudinal_pos: longitudinal position along road [m]
        :param lane_idx: lane index (0=rightmost, lanes_count-1=leftmost)
        :return: True if position is inside a construction zone
        """
        if not hasattr(self, 'construction_zones'):
            return False
            
        buffer = 20  # Additional buffer zone [m] - increased for safety
        
        for zone in self.construction_zones:
            # Check if longitudinal position is within zone (with buffer)
            if zone['start'] - buffer <= longitudinal_pos <= zone['end'] + buffer:
                    return True
        
        return False

    def _create_vehicles(self) -> None:
        """Create vehicles, avoiding construction zones."""
        import sys
        
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            # Create ego vehicle with retries to avoid construction zones
            max_attempts = 50
            for attempt in range(max_attempts):
                vehicle = Vehicle.create_random(
                    self.road,
                    speed=25.0,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                )
                
                # Check if vehicle is in construction zone
                lane = self.road.network.get_lane(vehicle.lane_index)
                long_pos = lane.local_coordinates(vehicle.position)[0]
                lane_idx = vehicle.lane_index[2]
                
                in_zone = self._is_in_forbidden_construction_zone(long_pos, lane_idx)
                if not in_zone:
                    break  # Valid position found
                else:
                    if attempt % 10 == 0:
                        print(f"[DEBUG] Ego spawn attempt {attempt+1}: long={long_pos:.1f}m, lane={lane_idx} (IN ZONE, retrying)")
                    if attempt == max_attempts - 1:
                        print(f"[WARNING] Couldn't find safe ego spawn after {max_attempts} attempts")
            
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            # Create other vehicles, avoiding construction zones
            vehicles_created = 0
            vehicles_skipped = 0
            for _ in range(others):
                max_attempts = 30
                for attempt in range(max_attempts):
                    vehicle = other_vehicles_type.create_random(
                        self.road, spacing=3.0  
                    )
                    
                    # Check if vehicle is in construction zone
                    lane = self.road.network.get_lane(vehicle.lane_index)
                    long_pos = lane.local_coordinates(vehicle.position)[0]
                    lane_idx = vehicle.lane_index[2]
                    
                    in_zone = self._is_in_forbidden_construction_zone(long_pos, lane_idx)
                    if not in_zone:
                        vehicle.randomize_behavior()
                        self.road.vehicles.append(vehicle)
                        vehicles_created += 1
                        break  # Valid position found
                    elif attempt == max_attempts - 1:
                        # Skip this vehicle if can't find valid position
                        vehicles_skipped += 1

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        print(rewards)
        reward = sum(
            #self.config.get(name, 1) * reward for name, reward in rewards.items()
            v for k,v in rewards.items()
        )
        '''if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )'''
        #reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        total_rewards = {}
        #neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        if self._is_in_construction_zone(self.vehicle.lane.local_coordinates(self.vehicle.position)[0]):
            forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            print('\nforward speed:',forward_speed)
            construction_min_speed = self.config['speed']['construction_zone_limit_mph'] - self.config['speed']['speed_tolerance_mph']
            construction_max_speed = self.config['speed']['construction_zone_limit_mph'] + self.config['speed']['speed_tolerance_mph']
            print(construction_max_speed)
            print(construction_min_speed)

            if forward_speed >= construction_min_speed and forward_speed <= construction_max_speed:
                total_rewards['speed_compliance'] = 0.25#self.config['reward']['speed_compliance']['within_limit']
            else:
                total_rewards['speed_compliance'] = -0.25#self.config['reward']['speed_violation']['beyond_limit']
        else:
            total_rewards['efficiency'] = 0.025

        if self._is_terminated():
            if self.vehicle.crashed:
                total_rewards['collision_reward'] = -50#self.config['safety_rules']['collision']['penalty']

        if self._is_truncated():
            print('???')
            total_rewards['end'] = 50

        '''if self._is_terminated() or self._is_truncated():
            total_rewards['progress'] = self.vehicle.lane.local_coordinates(self.vehicle.position)[0] / self.vehicle.lane.length'''

        '''longitudinal = self.vehicle.lane.local_coordinates(self.vehicle.position)[0]
        progress = longitudinal / self.vehicle.lane.length
        total_rewards['progress_reward'] = round(progress, 2)'''

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        total_rewards['high_speed_reward'] = np.clip(scaled_speed, 0, 0.5)
        '''return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }'''

        return total_rewards
            


class HighwayWithObstaclesEnvFast(HighwayWithObstaclesEnv):
    """
    A variant of highway-with-obstacles-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles and obstacles in the scene
        - fewer lanes, shorter episode duration
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "obstacles_count": 5,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        """Create vehicles using parent logic, then disable collision checks."""
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
