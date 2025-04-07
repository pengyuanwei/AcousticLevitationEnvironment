import math
import random
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple, Any, List, Dict, Type

from acousticlevitationenvironment.particles import particle_slim, target_slim
from acousticlevitationenvironment.utils import MultiAgentActionSpace, MultiAgentObservationSpace, create_points, optimal_pairing, check_and_correct_positions_fixed
from examples.utils.optimizer_utils_v2 import *
from examples.utils.path_smoothing_2 import *
from examples.utils.acoustic_utils_v2 import *


class PlannerAPFGorkov(gym.Env):
    """
    A multi-particle path planning environment for gymnasium.

    """
    def __init__(
            self,
            n_particles: int=8,
            target_radius: float=0.001,
            particle_radius: float=0.001,
            max_velocity: float=0.1,
            delta_time: float=1.0/10,
            max_timesteps: int=20,
            training_stage: int=1,
            levitator: Type['top_bottom_setup']=None
        ):

        self.n_particles = n_particles
        self.n_targets = n_particles
        self.target_radius = target_radius
        self.particle_radius = particle_radius
        self.delta_time = delta_time
        self.max_velocity = max_velocity
        self.time_step = 0
        self.max_timesteps = max_timesteps
        self.training_stage = training_stage
        self.levitator = levitator
        self.gorkov = []

        self.play_field_corners: Tuple[float, float, float, float, float, float] = (-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12)

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
    
    
    def input_start_end_points(self, start_points, target_points):
        self.start_positions = start_points
        self.target_positions = target_points


    def reset(self, seed=None, options=None):    
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

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

        # 保存TWGS参数
        positions = self.get_pos()
        gorkov, self.ref_in, self.ref_out = self.levitator.calculate_gorkov_wgs_single_state_v2(positions)
        print('Initial max Gorkov:', np.max(gorkov))

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
            # last_velocity, last_position, vX, vY, vZ, velocity, x, y, z
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
                
        reach_index = self.check_status(True)
        self.safety_area()
        if not np.all(self.collision == 0.0):
            print('APF: 模型生成的路径不满足距离约束，需要进行 path correction')
            self.APF_correction(reach_index)
        self.gorkov_correction(reach_index)
        self.update_dist()

        # 更新TWGS的 ref_in 和 ref_out
        positions = self.get_pos()
        last_positions = self.get_last_pos()
        positions = self.get_pos()
        _, _, _, paths, _ = uniform_velocity_interpolation_v2(
            start=last_positions, end=positions, total_time=self.delta_time, dt=0.0032, velocities=0.0
        )
        _, self.ref_in, self.ref_out = self.levitator.calculate_gorkov_twgs_input(paths, self.ref_in, self.ref_out)
        
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


    def APF_correction(self, reach_index):
        positions = self.get_pos()
        new_positions, satisfied = check_and_correct_positions_fixed(positions, reach_index)

        if satisfied:
            print('APF: success!')
            self.collision.fill(0.0)
            for i, particle in enumerate(self.particles):
                particle.x = new_positions[i][0]
                particle.y = new_positions[i][1]
                particle.z = new_positions[i][2]
            self.update_velocities()
        else:
            print('APF: 待 Gorkov optimization 尝试再次修正。')


    def gorkov_correction(self, reach_index):
        '''
        对第一个keypoint的修正
        '''
        positions = self.get_pos()
        last_positions = self.get_last_pos()
        # 在两个keypoints之间进行插值
        _, _, _, paths, _ = uniform_velocity_interpolation_v2(
            start=last_positions, end=positions, total_time=self.delta_time, dt=0.0032, velocities=0.0
        )
        # 计算插值后的所有轨迹的所有坐标中的最大gorkov
        gorkov, self.ref_in, self.ref_out = self.levitator.calculate_gorkov_twgs_input(paths, self.ref_in, self.ref_out)
        max_gorkov = np.max(gorkov)
        displacements = self.get_dis()
        last_displacements = self.get_last_dis()
        candidate_solutions, sorted_indices, sorted_solutions_max_gorkov = self.generate_solutions_single_frame(
            last_positions, positions, last_displacements, displacements, reach_index, num_solutions=50
        )

        if candidate_solutions is not None:
            print('Max Gorkov of original locations:', max_gorkov)
            print('Max Gorkov of the first candidate solution:', sorted_solutions_max_gorkov[0])
            # 依次取出 candidate_solutions，先检查是否Gorkov更好，再检查是否满足距离约束
            # 分别求出前后两个 segment 的最大位移，用于缩放时间
            re_plan_segment = np.concatenate([last_positions[np.newaxis, :, :], positions[np.newaxis, :, :]])

            for i in range(candidate_solutions.shape[1]):
                # 如果 candidate_solutions 的 Gorkov 比原坐标的更差，则 break
                if sorted_solutions_max_gorkov[i] > max_gorkov:
                    print('Gorkov optimization: no better candidate than original!')
                    break

                re_plan_segment[-1:, :, :] = np.transpose(
                    candidate_solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], 
                    (1, 0, 2)
                )

                # 通过插值检查是否碰撞
                interpolated_coords = interpolate_positions(re_plan_segment)
                for j in range(interpolated_coords.shape[0]):
                    collision = safety_area(self.n_particles, interpolated_coords[j])
                    if np.any(collision != 0):
                        break

                if np.all(collision == 0):
                    print(f"Gorkov optimization: final non-collision best candidate solution: No.{i}")
                    self.collision.fill(0.0)
                    for j, particle in enumerate(self.particles):
                        particle.x = candidate_solutions[j, sorted_indices[i], 0]
                        particle.y = candidate_solutions[j, sorted_indices[i], 1]
                        particle.z = candidate_solutions[j, sorted_indices[i], 2]
                    self.update_velocities()
                    break
        else:
            print("Gorkov optimization: 未能生成 candidate solutions, Gorkov 优化失败！")


    def update_dist(self):
        for i, particle in enumerate(self.particles):
            dist = math.sqrt((particle.x - self.targets[i].x)**2 + (particle.y - self.targets[i].y)**2 + (particle.z - self.targets[i].z)**2)
            particle.last_timestep_dist = dist
    
    def update_velocities(self):
        for particle in self.particles:
            delta_x = particle.x - particle.last_position[0]
            delta_y = particle.y - particle.last_position[1]
            delta_z = particle.z - particle.last_position[2]
            particle.vX = delta_x/self.delta_time
            particle.vY = delta_y/self.delta_time
            particle.vZ = delta_z/self.delta_time
            particle.velocity = np.array([particle.vX, particle.vY, particle.vZ])        
    

    def _is_it_terminated(self):
        return np.all(self.collision == 0.0) and all(particle.last_timestep_dist <= self.max_velocity*self.delta_time for particle in self.particles)


    def _is_it_truncated(self):        
        return np.any(self.collision != 0.0) or self.time_step >= self.max_timesteps


    def get_pos(self):
        positions = np.zeros((self.n_particles, 3))
        for i, particle in enumerate(self.particles):
            positions[i] = [particle.x, particle.y, particle.z]
        return positions
    
    def get_last_pos(self):
        positions = np.zeros((self.n_particles, 3))
        for i, particle in enumerate(self.particles):
            positions[i] = particle.last_position
        return positions

    def get_dis(self):
        displacements = np.zeros((self.n_particles, 3))
        for i, particle in enumerate(self.particles):
            displacements[i] = [particle.vX * self.delta_time, particle.vY * self.delta_time, particle.vZ * self.delta_time]
        return displacements
    
    def get_last_dis(self):
        displacements = np.zeros((self.n_particles, 3))
        for i, particle in enumerate(self.particles):
            displacements[i] = particle.last_velocity * self.delta_time
        return displacements
    
    def check_status(self, debug=False):
        reach_index = np.zeros((self.n_particles, 1))
        for i, particle in enumerate(self.particles):
            reach_index[i] = particle.reached_target
        if debug:
            print(f"Path finding: 第{self.time_step}时间步，剩余未到达终点的粒子数: {np.sum(reach_index == 0)}")
        return reach_index


    def generate_solutions_single_frame(
            self,
            last_positions: np.array, 
            positions: np.array, 
            last_displacements: np.array, 
            displacements: np.array, 
            reach_index: np.array,
            num_solutions: int=10
        ):
        '''
        为某个已知时刻生成solutions
        last_positions: (num_particles, 3)
        positions: (num_particles, 3)
        last_displacements: (num_particles, 3)
        displacements: (num_particles, 3)
        '''
        # 对最弱key points生成100个潜在solutions，并排序
        # if self.time_step > 1:
        #     candidate_solutions = create_constrained_points_single_frame(
        #         self.n_particles, 
        #         last_positions,
        #         positions,
        #         last_displacements, 
        #         displacements, 
        #         reach_index,
        #         num_solutions
        #     )
        # else:
        candidate_solutions = create_constrained_points_single_frame_v2(
            self.n_particles, 
            last_positions,
            positions,
            reach_index,
            num_solutions
        )
        if candidate_solutions is None:
            return None, None, None

        # 计算 candidate_solutions 的 Gorkov
        solutions_max_gorkov = np.zeros((candidate_solutions.shape[1], ))
        for i in range(candidate_solutions.shape[1]):
            _, _, _, paths, _ = uniform_velocity_interpolation_v2(
                start=last_positions, end=candidate_solutions[:, i, :], total_time=self.delta_time, dt=0.0032, velocities=0.0
            )
            solution_gorkov, _, _ = self.levitator.calculate_gorkov_twgs_input(paths, self.ref_in, self.ref_out)
            # 找出每个 candidate_solutions 的最大 Gorkov
            solutions_max_gorkov[i] = np.max(solution_gorkov)
            if i % 10 == 0:
                print(i)
        # 根据 max Gorkov 从小到大对 candidate_solutions 排序
        sorted_indices = np.argsort(solutions_max_gorkov)
        sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

        return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov