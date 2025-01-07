import math
import random
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple, Any, List, Dict

from acousticlevitationenvironment.envs import GlobalPlanner
from acousticlevitationenvironment.particles import particle_slim, target_slim
from acousticlevitationenvironment.utils import MultiAgentActionSpace, MultiAgentObservationSpace, create_points, create_points_multistage, optimal_pairing


class GlobalRePlanner(GlobalPlanner):
    """
    A multi-particle path planning environment for gymnasium.

    """
    def __init__(
            self,
            n_particles: int = 8,
            target_radius: float = 0.001,
            particle_radius: float = 0.001,
            max_velocity: float = 0.1,
            delta_time: float = 1.0/10,
            max_timesteps: int = 20,
            training_stage: int = 1
        ):

        super().__init__(
            n_particles,
            target_radius,
            particle_radius,
            max_velocity,
            delta_time,
            max_timesteps,
            training_stage
        )
    

    def input_start_end_points(self, start_points, target_points):
        self.start_positions = start_points
        self.target_positions = target_points


    def reset(self, seed=None, options=None):    
        # We need the following line to seed self.np_random
        gym.Env.reset(self, seed=seed)

        self.particles: List[particle_slim] = []
        self.targets: List[target_slim] = []

        self.particles_instancing(self.start_positions, self.target_positions)
            
        for i, particle in enumerate(self.particles):
            particle.inital_distance = math.sqrt((particle.x - self.targets[i].x)**2 
                                                + (particle.y - self.targets[i].y)**2 
                                                + (particle.z - self.targets[i].z)**2)
            particle.last_timestep_dist = particle.inital_distance    

        self.time_step = 0
        self.collision = np.zeros(self.n_particles)
    
        return self._get_obs(), self._info


    def particles_instancing(self, start_positions, target_positions):
        for i in range(self.n_particles):
            self.particles.append(particle_slim(start_positions[i][0], start_positions[i][1], start_positions[i][2]))
            self.targets.append(target_slim(target_positions[i][0], target_positions[i][1], target_positions[i][2]))