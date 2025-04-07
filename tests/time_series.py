import numpy as np

total_time = 0.0035
dt = 0.0032

t = np.arange(0, total_time, dt)
num_steps = len(t)

print(t)
print(num_steps)