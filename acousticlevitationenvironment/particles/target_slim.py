import math
import numpy as np


class Target:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.initPos = [x, y, z]


    def reset(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.initPos = [x, y, z]