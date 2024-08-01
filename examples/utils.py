import os
import csv
import math
import torch
import numpy as np
import gymnasium as gym

from acoustorl import MADDPG
from acousticlevitationgym.utils import general_utils
    

def interpolate_positions(coords, delta_time_original=0.2, delta_time_new=0.02):
    num_interpolations = int(delta_time_original / delta_time_new) - 1
    interpolated_coords = []

    for i in range(coords.shape[1]):
        start = coords[0, i]
        end = coords[1, i]
        
        # Calculate step for each dimension
        step = (end - start) / (num_interpolations + 1)
        
        # Generate interpolated positions
        positions = [start + j * step for j in range(num_interpolations + 2)]
        
        interpolated_coords.append(positions)
    
    # Convert list to numpy array
    interpolated_coords = np.array(interpolated_coords)
    
    # Reshape to match the required format (2 * (N * 10) / 10, N, 3)
    interpolated_coords = interpolated_coords.transpose(1, 0, 2).reshape(-1, coords.shape[1], 3)
    
    return interpolated_coords


def safety_area(n_particles, coords):
    collision = np.zeros(n_particles)
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

    for i in range(n_particles):
        x, y, z = [coords[i][0], coords[i][1], coords[i][2]]
        if not (x_min < x < x_max and y_min < y < y_max and z_min < z < z_max):
            collision[i] = 1.0
            
        for j in range(i+1, n_particles):
            dist = math.sqrt((x - coords[j][0])**2/0.014**2 + 
                                (y - coords[j][1])**2/0.014**2 + 
                                (z - coords[j][2])**2/0.03**2)
            if dist <= 1.0:
                collision[i] = 1.0
                collision[j] = 1.0

    return collision


def euclidean_distance_3d(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)


def calculate_3d_distances(coordinate_list):
    distances = []
    for sublist in coordinate_list:
        sublist_distances = []
        for i in range(len(sublist) - 1):
            dist = euclidean_distance_3d(sublist[i], sublist[i+1])
            sublist_distances.append(dist)
        distances.append(sublist_distances)
    return distances


def generate_global_paths(env, agent, n_particles, max_timesteps):
    paths = [[] for _ in range(n_particles)]
        
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = np.zeros((n_particles, 3))
    final_points = state[:, :3] + state[:, 6:9]

    for _ in range(max_timesteps):
        action = agent.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or truncated:
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated


def generate_replan_paths(env, agent, n_particles, max_timesteps, points):
    paths = [[] for _ in range(n_particles)]

    start_points = points[0]
    target_points = points[-1]
    
    env.unwrapped.input_start_end_points(start_points, target_points)
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = np.zeros((n_particles, 3))
    final_points = state[:, :3] + state[:, 6:9]

    for _ in range(max_timesteps):
        action = agent.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or truncated:
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated