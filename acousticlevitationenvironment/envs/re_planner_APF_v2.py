import math
import random
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple, Any, List, Dict

from acousticlevitationenvironment.envs import PlannerAPF
from acousticlevitationenvironment.particles import particle_slim, target_slim
from acousticlevitationenvironment.utils import check_and_correct_positions


class RePlannerAPFv2(PlannerAPF):
    """
    APF: 没有固定的粒子

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


    def step(self, action):
        self.collision = np.zeros(self.n_particles)
        reward = np.zeros(self.n_particles)
        self.time_step += 1

        for i, particle in enumerate(self.particles):
            particle.last_velocity = np.array([particle.vX, particle.vY, particle.vZ])
            particle.last_position = [particle.x, particle.y, particle.z]

            if particle.last_timestep_dist > self.max_velocity*self.delta_time:
                particle.vX = action[i][0]
                particle.vY = action[i][1]
                particle.vZ = action[i][2]                  
                particle.velocity = np.array([particle.vX, particle.vY, particle.vZ])
                particle.x = min(max(particle.x + particle.vX * self.delta_time, self.play_field_corners[0]), self.play_field_corners[1])
                particle.y = min(max(particle.y + particle.vY * self.delta_time, self.play_field_corners[2]), self.play_field_corners[3])
                particle.z = min(max(particle.z + particle.vZ * self.delta_time, self.play_field_corners[4]), self.play_field_corners[5])
                particle.reached_target = False
            elif self.particle_radius < particle.last_timestep_dist <= self.max_velocity*self.delta_time:
                delta_x = self.targets[i].x - particle.x
                delta_y = self.targets[i].y - particle.y
                delta_z = self.targets[i].z - particle.z
                particle.vX = delta_x/self.delta_time
                particle.vY = delta_y/self.delta_time
                particle.vZ = delta_z/self.delta_time
                particle.velocity = np.array([particle.vX, particle.vY, particle.vZ])
                particle.x = self.targets[i].x
                particle.y = self.targets[i].y
                particle.z = self.targets[i].z
                particle.reached_target = True
            else:
                particle.vX = 0.0
                particle.vY = 0.0
                particle.vZ = 0.0
                particle.velocity = np.array([particle.vX, particle.vY, particle.vZ])
                particle.x = self.targets[i].x
                particle.y = self.targets[i].y
                particle.z = self.targets[i].z     
                particle.reached_target = True
                
        _ = self.check_status(True)
        self.safety_area()
        if not np.all(self.collision == 0.0):
            print('APF: 模型生成的路径不满足距离约束，需要进行 path correction')
            self.APF_correction()
        self.update_dist()

        terminated = self._is_it_terminated()
        truncated = self._is_it_truncated()

        return self._get_obs(), reward, terminated, truncated, self._info


    def safety_area(self):
        for i in range(self.n_particles):
            x, y, z = [self.particles[i].x, self.particles[i].y, self.particles[i].z]
            for j in range(i+1, self.n_particles):
                dist_square = (x - self.particles[j].x)**2/0.014**2 + (y - self.particles[j].y)**2/0.014**2 + (z - self.particles[j].z)**2/0.03**2
                if dist_square <= 1.0:
                    self.collision[i] = 1.0
                    self.collision[j] = 1.0


    def APF_correction(self):
        positions = self.get_pos()
        new_positions, satisfied = check_and_correct_positions(positions)

        if satisfied:
            print('APF: success!')
            self.collision.fill(0.0)
            for i, particle in enumerate(self.particles):
                particle.x = new_positions[i][0]
                particle.y = new_positions[i][1]
                particle.z = new_positions[i][2]
        else:
            print('APF: failure!')


    def update_dist(self):
        for i, particle in enumerate(self.particles):
            dist = math.sqrt((particle.x - self.targets[i].x)**2 + (particle.y - self.targets[i].y)**2 + (particle.z - self.targets[i].z)**2)
            particle.last_timestep_dist = dist


    def _is_it_terminated(self):
        return np.all(self.collision == 0.0) and all(particle.last_timestep_dist <= self.max_velocity*self.delta_time for particle in self.particles)


    def _is_it_truncated(self):        
        return np.any(self.collision != 0.0) or self.time_step >= self.max_timesteps
    

    def get_pos(self):
        positions = np.zeros((self.n_particles, 3))
        for i, particle in enumerate(self.particles):
            positions[i] = [particle.x, particle.y, particle.z]
        return positions
    

    def check_status(self, debug=False):
        reach_index = np.zeros((self.n_particles, 1))
        for i, particle in enumerate(self.particles):
            reach_index[i] = particle.reached_target
        if debug:
            print(f"Path finding: 第{self.time_step}时间步，剩余未到达终点的粒子数: {np.sum(reach_index == 0)}")
        return reach_index