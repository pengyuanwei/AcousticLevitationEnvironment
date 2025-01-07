import math
import random
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple, Any, List, Dict

from acousticlevitationenvironment.particles import particle_slim, target_slim
from acousticlevitationenvironment.utils import MultiAgentActionSpace, MultiAgentObservationSpace, create_points, create_points_multistage, optimal_pairing


class GlobalTrain(gym.Env):
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

        self.n_particles = n_particles
        self.n_targets = n_particles
        self.target_radius = target_radius
        self.particle_radius = particle_radius
        self.delta_time = delta_time
        self.max_velocity = max_velocity
        self.time_step: int = 0
        self.max_timesteps = max_timesteps
        self.training_stage = training_stage

        self.play_field_corners: Tuple[float, float, float, float, float, float] = (-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12)

        self.centerPos = [0.00, 0.00, 0.12]

        self.angle = math.pi/self.n_particles

        self.action_space = MultiAgentActionSpace([gym.spaces.Box(-max_velocity, max_velocity, shape=(3,), dtype=np.float32) for _ in range(self.n_particles)])
        
        self.observation_dim = 3*(n_particles+2)
        self.observation_space = MultiAgentObservationSpace([gym.spaces.Box(-np.inf, np.inf, shape=(self.observation_dim,), dtype=np.float32)] * self.n_particles)

        self._info = {
            "n_particles": self.n_particles,
            "max_velocity": self.max_velocity,
            "target_radius": self.target_radius,
            "particle_radius": self.particle_radius,
            "delta_time": self.delta_time,
            "satisfy_constraints": True
        }


    def _get_obs(self):
        observation = np.zeros((self.n_particles, self.observation_dim))
        for i, particle in enumerate(self.particles):
            other_particle = []
            for _particle in self.particles:
                if _particle != particle:
                    _other_particle = [_particle.x - particle.x,
                                       _particle.y - particle.y,
                                       _particle.z - particle.z]
                    other_particle.extend(_other_particle)

            obs = [particle.x, 
                   particle.y, 
                   particle.z,
                   particle.vX, 
                   particle.vY, 
                   particle.vZ,
                   self.targets[i].x - particle.x,
                   self.targets[i].y - particle.y,
                   self.targets[i].z - particle.z]
                
            observ = np.concatenate((obs, other_particle), axis=0)
            observation[i, :] = observ

        return observation


    def reset(self, seed=None, options=None):    
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.particles: List[particle_slim] = []
        self.targets: List[target_slim] = []

        start_positions = create_points(self.n_particles)
        target_positions = create_points(self.n_particles)
        
        # 计算最优配对
        pairs = optimal_pairing(start_positions, target_positions)

        self.particles_instancing(start_positions, target_positions, pairs)
            
        for i, particle in enumerate(self.particles):
            particle.inital_distance = math.sqrt((particle.x - self.targets[i].x)**2 
                                                + (particle.y - self.targets[i].y)**2 
                                                + (particle.z - self.targets[i].z)**2)
            particle.last_timestep_dist = particle.inital_distance    

        self.time_step = 0
        self.collision = np.zeros(self.n_particles)
    
        return self._get_obs(), self._info


    def particles_instancing(self, start_positions, target_positions, pairs):
        for i in range(self.n_particles):
            self.particles.append(particle_slim(start_positions[i][0], start_positions[i][1], start_positions[i][2]))
            index = pairs[i][1]
            self.targets.append(target_slim(target_positions[index][0], target_positions[index][1], target_positions[index][2]))
    

    def expert_actions(self):
        actions = np.zeros((self.n_particles, 3))
        for i, particle in enumerate(self.particles):
            action = [particle.vX, 
                      particle.vY, 
                      particle.vZ]
                
            actions[i, :] = action

        return actions
    
    
    def step(self, action):
        self.collision = np.zeros(self.n_particles)
        reward = np.zeros(self.n_particles)
        
        self.time_step += 1

        for i, particle in enumerate(self.particles):
            particle.last_velocity = np.array([particle.vX, particle.vY, particle.vZ])
            particle.last_position = [particle.x, particle.y, particle.z]

            if particle.last_timestep_dist > 0.02:
                particle.vX = action[i][0]
                particle.vY = action[i][1]
                particle.vZ = action[i][2]
            elif self.max_velocity*self.delta_time < particle.last_timestep_dist <= 0.02:
                # 生成0到1之间的随机数
                if random.random() < 0.5:
                    delta_x = self.targets[i].x - particle.x
                    delta_y = self.targets[i].y - particle.y
                    delta_z = self.targets[i].z - particle.z
                    max_abs_num = self.find_max_abs(delta_x, delta_y, delta_z)
                    if max_abs_num != 0.0:
                        particle.vX = (delta_x/max_abs_num)*self.max_velocity
                        particle.vY = (delta_y/max_abs_num)*self.max_velocity
                        particle.vZ = (delta_z/max_abs_num)*self.max_velocity
                    else:
                        particle.vX = 0.0
                        particle.vY = 0.0
                        particle.vZ = 0.0
                else:
                    particle.vX = action[i][0]
                    particle.vY = action[i][1]
                    particle.vZ = action[i][2]                    
            else:
                if random.random() < 0.5:
                    delta_x = self.targets[i].x - particle.x
                    delta_y = self.targets[i].y - particle.y
                    delta_z = self.targets[i].z - particle.z
                    particle.vX = delta_x/self.delta_time
                    particle.vY = delta_y/self.delta_time
                    particle.vZ = delta_z/self.delta_time
                else:
                    particle.vX = action[i][0]
                    particle.vY = action[i][1]
                    particle.vZ = action[i][2] 

            particle.velocity = np.array([particle.vX, particle.vY, particle.vZ])

        for i, particle in enumerate(self.particles): 
            particle.x = particle.x + particle.vX * self.delta_time
            particle.y = particle.y + particle.vY * self.delta_time
            particle.z = particle.z + particle.vZ * self.delta_time

        self.safety_area()

        # 初始化 multiplicative_factor 为全 1 的数组
        multiplicative_factor = np.ones(self.n_particles)
        # 将发生碰撞的粒子的 multiplicative_factor 设为 0
        multiplicative_factor[self.collision != 0.0] = 0.0
        # 计算临时奖励
        temp = self._calculate_reward()
        # 根据 multiplicative_factor 调整 temp
        temp *= multiplicative_factor
        # 计算每个粒子的奖励
        reward = (1 - self.collision) * np.sum(temp)

        if_terminated = self._is_it_terminated()
        if if_terminated == 1:
            reward += 20.0
            terminated = True
        else:
            terminated = False

        truncated = self._is_it_truncated()

        return self._get_obs(), reward, terminated, truncated, self._info
    

    def safety_area(self):
        x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

        for i in range(self.n_particles):
            x, y, z = [self.particles[i].x, self.particles[i].y, self.particles[i].z]
            if not (x_min < x < x_max and y_min < y < y_max and z_min < z < z_max):
                self.collision[i] = 1.0
                
            for j in range(i+1, self.n_particles):
                dist = math.sqrt((x - self.particles[j].x)**2/0.014**2 + 
                                 (y - self.particles[j].y)**2/0.014**2 + 
                                 (z - self.particles[j].z)**2/0.03**2)
                if dist <= 1.0:
                    self.collision[i] = 1.0
                    self.collision[j] = 1.0
    

    def _calculate_reward(self):

        reward = self.reward_function_1()
        
        return reward
    

    def reward_function_1(self):
        reward = np.zeros(self.n_particles)

        for i, particle in enumerate(self.particles):
            dist = math.sqrt((particle.x - self.targets[i].x)**2 + (particle.y - self.targets[i].y)**2 + (particle.z - self.targets[i].z)**2)
            if dist < self.particle_radius:
                reward[i] += 1.0
            elif dist < 2*self.particle_radius:
                reward[i] += (1.36 + 300*(2*self.particle_radius - dist))/1.96
            elif dist < 10*self.particle_radius:
                reward[i] += (0.4 + 120*(10*self.particle_radius - dist))/1.96
            else:
                reward[i] += (0.1732 + (particle.last_timestep_dist - dist)/self.delta_time)/1.96
            particle.last_timestep_dist = dist

        return reward


    def find_max_abs(self, num1, num2, num3):
        abs_num1 = abs(num1)
        abs_num2 = abs(num2)
        abs_num3 = abs(num3)
        
        return max(abs_num1, abs_num2, abs_num3)


    def _is_it_terminated(self):
        terminated = 1

        for i in range(len(self.particles)):
            if self.particles[i].last_timestep_dist > (2*self.particle_radius):
                terminated = 0
                return terminated

        return terminated


    def _is_it_truncated(self):
        truncated = False

        if self.time_step >= self.max_timesteps:
            truncated = True
        
        return truncated