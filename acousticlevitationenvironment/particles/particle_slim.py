import math
import numpy as np


class Particle:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.initPos = [x, y, z]
        self.last_position = [x, y, z]       
        self.vX = 0.0
        self.vY = 0.0
        self.vZ = 0.0
        self.velocity = np.array([self.vX, self.vY, self.vZ])
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        self.target = 0
        self.inital_distance = 0.0
        self.last_timestep_dist = 0.0
        self.reached_target = False
        self.collision = 0
        self.min_dist_with_others = 1.0
        self.v = 0.0
        self.delta_t = 0.0


    def reset(self):
        [self.x, self.y, self.z] = self.initPos
        self.last_position = [self.initPos[0], self.initPos[1], self.initPos[2]]
        self.vX = 0.0
        self.vY = 0.0
        self.vZ = 0.0
        self.velocity = np.array([self.vX, self.vY, self.vZ])
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        self.target = 0
        self.inital_distance = 0.0
        self.last_timestep_dist = 0.0
        self.reached_target = False
        self.collision = 0
        self.min_dist_with_others = 1.0
        self.v = 0.0
        self.delta_t = 0.0


    def random_reset(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.initPos = [x, y, z] 
        self.last_position = [x, y, z]
        self.vX = 0.0
        self.vY = 0.0
        self.vZ = 0.0
        self.velocity = np.array([self.vX, self.vY, self.vZ])
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        self.target = 0
        self.inital_distance = 0.0
        self.last_timestep_dist = 0.0
        self.reached_target = False
        self.collision = 0
        self.min_dist_with_others = 1.0
        self.v = 0.0
        self.delta_t = 0.0